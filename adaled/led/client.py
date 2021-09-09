from adaled.backends import cmap, get_backend, TensorCollection
from adaled.conduits.base import Conduit
from adaled.led.criteria import AdaLEDCriteria
from adaled.led.diagnostics import AdaLEDStage, AdaLEDStep, AdaLEDCycleDiagnostics
from adaled.led.recording import RecorderConfig, TrajectoryRecorder
from adaled.led.topology import Topology
from adaled.solvers.base import MicroSolver, MacroSolver
from adaled.transformers.base import Transformer, IdentityTransformer
from adaled.utils.buffer import DynamicArray
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.misc import batch_mse
from adaled.utils.mpi import cpu_friendly_barrier
import adaled

import numpy as np
import torch

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Sequence, Union
import dataclasses
import os
import time

NOT_AVAILABLE = object()

# TODO: Refactor the client code. Although the distinction between Client and
# Generator/Runner kind of makes sense (Client talks to the Server, it does not
# run the simulation), Generator and Runner should probably be merged.

# TODO: Move Client.reject_until_timestep to Criteria. Make Criteria stateful,
#       i.e. make Client aware of that possibility. Provide more information to
#       Criteria, like the current time step, possibly using struct-like
#       arguments instead of passing a gazillion of them manually.

@dataclass
class AdaLEDClientConfig(DataclassMixin):
    """Configuration of AdaLED clients (inference)."""
    # Total number of simulation steps. 0 means infinity.
    max_steps: int = 0

    # Every how many steps to print the step on the screen. 0 to disable.
    log_every: int = 0

    # Always run the micro solver, even in the macro stage. These states are
    # *not* taken into account when building the trajectory dataset.
    # For expensive simulations, consider setting server.sleep_micro_on_macro=
    always_run_micro: bool = False

    # Compute the validation error on `update_state`. Available only when
    # `always_run_micro` is `True`.
    compute_validation_error: bool = True

    # Until which time step to reject all cycles.
    reject_until_timestep: int = 0

    # Profile the execution time of functions with cProfile.
    profile: bool = False

    # Reduce verbosity.
    quiet: bool = False


def _expand_forcing(F, like):
    batch_size = len(like)
    if len(F) != batch_size:
        raise TypeError(f"length of external forcing should match the batch "
                        f"size of {batch_size}: F.shape={F.shape} "
                        f"like.shape={like.shape}")
    # Be strict about `F`, do not automatically broadcast as this may bring
    # havoc later on.
    return F


def merge_real_and_forcing(x, F):
    """Merge real state and forcing state into a TensorCollection(x=x, F=F).

    If `F` is a float, it is expanded to a batch of values.
    """
    return TensorCollection(x=x, F=_expand_forcing(F, x))


def merge_latent_and_forcing(z, F):
    """Merge real state and forcing state into a TensorCollection(z=z, F=F).

    If `F` is a float, it is expanded to a batch of values.
    """
    return TensorCollection(z=z, F=_expand_forcing(F, z))


class AdaLEDClient:
    """Client side of AdaLED.

    Receives the macro solver and the transformer from the server, and sends
    diagnostics and newly gathered trajectories back.
    """
    def __init__(self,
                 config: AdaLEDClientConfig,
                 macro: MacroSolver,
                 criteria: AdaLEDCriteria,
                 transformer: Transformer = IdentityTransformer(),
                 *,
                 comm: Optional['mpi4py.MPI.Intracomm'] = None,
                 topology: Optional[Topology] = None):
        """
        Use the `topology` argument when working with distributed setups, where
        each server and client runs on its own ranks. For setups where each
        rank runs both the server and the client, use the `comm` argument or,
        better, use the `AdaLED` class.
        """
        if comm is not None and topology is not None:
            raise TypeError("cannot specify both `comm` and `topology`")
        self.config = config
        self.macro = macro
        self.transformer = transformer
        self.criteria = criteria
        self.topology = topology
        self.active_client_comm = comm and topology and topology.active_side_comm
        self.start_time = time.time()

        self._save_restart_to: Optional[str] = None
        self._restart_from: Optional[str] = None
        self._received_server_data = False

    def make_generator(
            self,
            micro: MicroSolver,
            external_forcing: Optional[Iterable[Any]] = None):
        """Run AdaLED on the given micro solver and with the given external
        forcing.

        Returns an infinite generator of AdaLED steps.
        """
        return AdaLEDGenerator(self, micro, external_forcing)

    def run(self, *args, **kwargs):
        runner = self.make_runner(*args, **kwargs)
        runner.run()

    def make_runner(
            self,
            generator: 'AdaLEDGenerator',
            recorder: Union[TrajectoryRecorder, RecorderConfig] = RecorderConfig(),
            recorder_cls: type = TrajectoryRecorder,
            **kwargs):
        """Create a runner for the given generator.

        In the future, may be extended to support multiple generators."""
        topology = self.topology
        if isinstance(recorder, RecorderConfig):
            # For now we assume there is only one active rank per simulation
            # (the root rank).
            sim_id = self.active_client_comm.rank if self.active_client_comm else 0
            recorder = recorder_cls(recorder, sim_id=sim_id)
        if topology:
            return _ClientRunner(
                    self, generator, recorder, topology.active_intercomm,
                    topology.remote_activecomm_ranks, **kwargs)
        else:
            return _ClientRunner(self, generator, recorder, None, [0], **kwargs)

    def process_server_data(self, data: dict):
        print("Processing server data.")
        if 'restart_from' in data:
            self._restart_from = data['restart_from']
        if 'save_restart_to' in data:
            self._save_restart_to = data['save_restart_to']
        self._received_server_data = True
        self.macro.load_state_dict(data['macro'])
        self.transformer.load_state_dict(data['transformer'])

    def get_latest_networks(self):
        return self.macro, self.transformer


class AdaLEDGenerator:
    def __init__(self,
                 parent: AdaLEDClient,
                 micro: MicroSolver,
                 external_forcing: Optional[Iterable[Any]]):
        self.parent = parent
        self.micro = micro
        if external_forcing is None:
            batch_size = len(self.micro.get_state())
            _empty = np.zeros((batch_size, 0))
            def empty_external_forcing():
                while True:
                    yield _empty
            external_forcing = empty_external_forcing()
        self.external_forcing = external_forcing

        self._new_trajectories = DynamicArray()
        self._cycle_diagnostics = AdaLEDCycleDiagnostics()
        self._F = NOT_AVAILABLE
        self._timestep = 0
        self._cycle_data = None

        # Overhead in seconds spent on expensive analysis that are not
        # otherwise present in production runs.
        self.pending_stats_overhead = 0.0
        self._overhead_counter = 0  # Counts nested scopes.

    def state_dict(self) -> dict:
        return {
            'micro': self.micro.state_dict(),  # Collective operation.
            'timestep': self._timestep,
        }

    def load_state_dict(self, state: dict):
        self.micro.load_state_dict(state['micro'])
        self._timestep = state['timestep']

        # Alternatively, we could store external forcing as part of the state.
        # This seems to be more flexible at the moment.
        for i in range(self._timestep):
            self._next_forcing()

    @contextmanager
    def measure_stats_overhead(self):
        """A context manager that measures the execution time of the
        `with` block. If `server.compensate_for_stats_overhead` is set to
        `True`, the total time spent in these blocks will be compensated by the
        server by pausing the training for the corresponding time.

        Used to ensure analysis/stats/debug runs have qualitatively the same performance
        as the production runs, i.e. that added stats overhead does not
        positively affect the training. Should be used sparingly, e.g. only
        for parts of the code related to `client.always_run_micro == True`.

        Supports nested blocks (only the outermost will be counted).

        Usage:
        >>> with cycle.measure_stats_overhead():
                something_potentially_expensive()
        """
        t0 = time.time()
        self._overhead_counter += 1
        try:
            yield None
        finally:
            self._overhead_counter -= 1
            if self._overhead_counter == 0:
                self.pending_stats_overhead += time.time() - t0

    def get_data_for_server(self):
        assert self._cycle_data is not None
        return self._cycle_data

    def start_cycle(self) -> Iterator[AdaLEDStep]:
        """Start an AdaLED cycle, returns an iterator of on-demand AdaLED steps."""
        if self.parent.config.profile:
            from cProfile import Profile
            profile = Profile()
            profile.enable()
        else:
            profile = None

        try:
            yield from self._do_cycle()
        finally:
            if profile:
                profile.disable()
                profile.print_stats(sort='tottime')

    def _next_forcing(self):
        return next(self.external_forcing)

    def _do_cycle(self):
        """Yield one AdaLED cycle, measure diagnostics and trajectories."""
        self._cycle_diagnostics = AdaLEDCycleDiagnostics()

        x = self.micro.get_state()
        F = self._next_forcing() if self._F is NOT_AVAILABLE else self._F
        def _assert_ndim(x_):
            if x_.ndim < 2:
                raise TypeError(f"expected a tensor of shape (batch, state...), got `{x_.shape}`.")
        adaled.cforeach(_assert_ndim, x)

        self._new_trajectories.clear()
        self._new_trajectories.append(merge_real_and_forcing(x, F))

        # To avoid race condition, read the reference to macro solver, config,
        # transformer and other volatile objects before starting the cycle. If
        # the parent updates the RNNs or the transformer concurrently, it
        # will/should replace the whole object (its reference), such that
        # cycles don't partially run with old partially with the new networks
        # and setup. (Other volatile objects are loaded in _do_cycle_steps.)
        macro, transformer = self.parent.get_latest_networks()

        # Immediately set to evaluation mode.
        macro.train(False)
        model: Optional[torch.nn.Module] = getattr(transformer, 'model', None)
        if model:
            model.train(False)

        start_timestep = self._timestep
        for step in self._do_cycle_steps(macro, transformer, x, F):
            if step.x is not None \
                    and step.stage != AdaLEDStage.MACRO \
                    and step.stage != AdaLEDStage.RELAXATION:
                self._new_trajectories.append(merge_real_and_forcing(step.x, step.F))
            yield step
            self._timestep += 1

        self._cycle_diagnostics.total_steps = self._timestep - start_timestep
        self._cycle_diagnostics.stats_overhead_seconds = self.pending_stats_overhead
        self.pending_stats_overhead = 0.0
        self._update_cycle_data(start_timestep, macro)

    def _update_cycle_data(self, start_timestep: int, macro: MacroSolver):
        """Finalize the gathered trajectory and store in self._cycle_data."""
        traj_len = len(self._new_trajectories)

        # Move the batch dimension to the beginning. Make a copy in order to be
        # able to send it before _new_trajectories.data is overwritten.
        def _copy_and_transpose(x):
            backend = get_backend(x)
            x = backend.detach(x)  # Just in case.
            x = backend.moveaxis(x, 0, 1)
            x = backend.clone(x)
            return x

        new_trajectories = cmap(_copy_and_transpose, self._new_trajectories.data)

        num_traj = len(new_trajectories)
        def broadcast(a):
            a = np.asarray(a)
            return np.broadcast_to(a, (num_traj, traj_len) + a.shape)

        decorated_trajectories = TensorCollection({
            'trajectory': new_trajectories,
            # Broadcast metadata along the batch and trajectory immediately.
            # Otherwise, working with as_trajectories and especially as_states
            # becomes a total mess.
            'metadata': {
                't': np.broadcast_to(
                    np.arange(start_timestep, start_timestep + traj_len),
                    (num_traj, traj_len)),
                'macro': cmap(broadcast, macro.metadata),
                # cmp_error and cmp_uncertainty added on the server side (see
                # AdaLEDServer.process_client_data), as they are already passed
                # under cycle diagnostics. Server side further adds other
                # properties like transformer and macro losses.
            },
        })
        # FIXME: Handle cycle diagnostics on the client side, not on the server
        # side. Sending cycle diagnostics also delays the sending of the
        # trajectory because we have to wait for the macro and the relaxation
        # phases to finish. Note however that certain diagnostics are stored to
        # the dataset together with the trajectories (see server's
        # decorate_trajectories).
        self._cycle_data = (decorated_trajectories, self._cycle_diagnostics)

    def _do_cycle_steps(self, macro, transformer, x, F):
        """Perform the AdaLED cycle algorithm.

        The algorithm includes:
            1) warmup
            2) comparison (evaluation of the macro solver)
            3a) macro evolution (if macro error is low)
            3b) relaxation (post-macro)
            4) micro evolution (if macro error is high)
        """
        # Note: Be careful with `torch.no_grad()` and `yield`, do not put
        # `yield` within `torch.no_grad()` as it would affect the caller!

        # Load potentially volatile attributes in advance, to ensure they are
        # consistent during the whole cycle.
        micro = self.micro
        cycle = self._cycle_diagnostics
        criteria = self.parent.criteria
        always_run_micro = self.parent.config.always_run_micro

        cycle.start_timestep = self._timestep

        # Warmup phase: feed the RNN with the micro results.
        z = h = None
        step = 0
        while criteria.should_continue_warmup(step, x, F, z, h, transformer):
            with torch.no_grad():
                # Pass the old x, not the new one!
                z = torch.as_tensor(transformer.transform(x))
                x = micro.advance(F)
                z_next, h, uncertainty = macro.advance(
                        merge_latent_and_forcing(z, F), h,
                        teacher_forcing=True)
            yield AdaLEDStep(AdaLEDStage.WARMUP, x, z_next, F, transformer,
                             uncertainty, h)
            step += 1
            F = self._next_forcing()

        # Evaluation phase: run both independently and compute macro's accuracy
        with torch.no_grad():
            z = torch.as_tensor(transformer.transform(x))
        step = 0
        while True:
            is_final = criteria.should_end_comparison(
                    step, x, F, z, h, transformer)
            x = micro.advance(F)
            with torch.no_grad():
                # Note: at step 0 we still have teacher forcing enabled,
                # otherwise the last step of warmup would be ignored.
                z, h, uncertainty = macro.advance(
                        merge_latent_and_forcing(z, F), h,
                        compute_uncertainty=is_final,
                        teacher_forcing=(step == 0))
            yield AdaLEDStep(AdaLEDStage.COMPARISON, x, z, F, transformer,
                             uncertainty, h)
            step += 1
            F = self._next_forcing()
            if is_final:
                break

        cycle.cmp_uncertainty = uncertainty
        cycle.cmp_error, accepted = criteria.compute_and_check_error(
                uncertainty, x, F, z, h, transformer)

        # Run macro if considered accurate enough.
        step = 0
        while accepted and self._timestep > self.parent.config.reject_until_timestep:
            with torch.no_grad():
                next_z, h, uncertainty = macro.advance(
                        merge_latent_and_forcing(z, F), h,
                        compute_uncertainty=True)
            # TODO: should_accept_macro_step should be able to return that
            # this is final step to accept. Now, in case of limited number
            # of steps, we nevertheless compute the final step only to
            # reject it. Return an enum (REJECT, ACCEPT, ACCEPT_FINAL)?
            if criteria.should_accept_macro_step(
                    step, uncertainty, F, next_z, h, transformer):
                z = next_z
                if always_run_micro:
                    with self.measure_stats_overhead():
                        x = micro.advance(F)
                else:
                    x = None
                yield AdaLEDStep(AdaLEDStage.MACRO, x, z, F,
                                 transformer, uncertainty, h)
                step += 1
            else:
                # Macro stop criterion met, cancel current time step and go
                # back to micro.
                if step > 0:
                    with torch.no_grad():
                        x = transformer.inverse_transform(z)
                    if self.parent.config.always_run_micro \
                            and self.parent.config.compute_validation_error:
                        with self.measure_stats_overhead():
                            x_expected = micro.get_state()
                            # For now assuming only xs are needed.
                            cycle.validation_error = criteria.compute_error(x, x_expected)
                            del x_expected

                    micro.update_state(new_state=x,
                                       skip_steps=(0 if always_run_micro else step))
                break
            F = self._next_forcing()
        cycle.macro_steps = step

        # Run relaxation if macro was used, otherwise micro.
        if step > 0:
            step = 0
            while criteria.should_continue_relaxation(step, x, F):
                x = micro.advance(F)
                yield AdaLEDStep(AdaLEDStage.RELAXATION, x, None, F,
                                 transformer, None, None)
                step += 1
                F = self._next_forcing()
        else:
            # step = 0  <-- already zero
            while criteria.should_continue_micro(step, x, F):
                x = micro.advance(F)
                yield AdaLEDStep(AdaLEDStage.MICRO, x, None, F, transformer,
                                 None, None)
                step += 1
                F = self._next_forcing()

        # This could be in principle returned and then catched with a
        # try..except block.
        self._F = F


class _ClientRunner:
    def __init__(self,
                 client: AdaLEDClient,
                 generator: AdaLEDGenerator,
                 recorder: TrajectoryRecorder,
                 intercomm,
                 server_ranks: Sequence[int],
                 conduit: Optional[Conduit] = None):
        assert generator
        assert len(server_ranks) > 0
        self.client = client
        self.generator = generator
        self.recorder = recorder
        self.intercomm = intercomm
        self._step = 0

        if not conduit:
            if intercomm:
                from adaled.conduits.mpi import IntercommConduit
                conduit = IntercommConduit(intercomm, max_messages=1)
            else:
                raise TypeError("conduit must be set manually in non-mpi runs")
        self.conduit = conduit
        self.server_ranks = server_ranks
        self._next_server = 0  # Cycle through ranks.

    def __del__(self):
        if self.conduit.active():
            import warnings
            warnings.warn("the client runner was not closed")
            self.close()

    def state_dict(self) -> dict:
        return {
            'generator': self.generator.state_dict(),  # Potentially expensive.
            'recorder': self.recorder.state_dict(),    # Potentially expensive.
            'step': self._step,  # Should be equal to generator._timestep.
            'next_server': self._next_server,
        }

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state['generator'])
        self.recorder.load_state_dict(state['recorder'])
        self._step = state['step']
        self._next_server = state['next_server'] % len(self.server_ranks)

    def save(self, dir: str):
        """Save the micro solver state."""
        client = self.client
        state = self.state_dict()  # Collective operation.

        if not client.active_client_comm or client.active_client_comm.rank == 0:
            path = os.path.join(dir, f'client-{self.recorder.sim_id:03d}.pt')
            print(f"Saving client runner state to {path}.")
            adaled.save(state, path)

    def restart(self, dir: str):
        path = os.path.join(dir, f'client-{self.recorder.sim_id:03d}.pt')
        print(f"Restarting client runner and generator from {path}.")
        state = adaled.load(path)
        self.load_state_dict(state)

    def start(self):
        if not self.client.config.quiet:
            print("Starting AdaLED client.", flush=True)
        self.conduit.start()

    def run(self):
        self.start()
        try:
            while True:
                done = self.run_cycle()
                if done:
                    self.close()
                    break
        except:
            if self.conduit:
                self.conduit.abort()
            raise
        print("Client execution time:", time.time() - self.client.start_time)

    def run_cycle(self) -> bool:
        """Run the cycle and return whether the simulation should be stopped."""
        messages = self.conduit.pop_all()  # Max 1 iteration.
        if not self.client._received_server_data:
            # Wait for the package from the server before running anything.
            while not messages:
                time.sleep(1)
                messages = self.conduit.pop_all()  # Max 1 iteration.

        for message in messages:
            self.client.process_server_data(message.data)

        if self.client._restart_from:
            self.restart(self.client._restart_from)
            # FIXME: refactor client/generator/runner, there are too many classes.
            self.client._restart_from = None

        for step in self.generator.start_cycle():
            i = self._step
            self._step += 1
            self.process_step(i, step)
            if self._is_done(i):
                return True

        cycle_data = self.generator.get_data_for_server()
        print(f"Sending cycle data (trajectory length of "
              f"{len(cycle_data[0][0])}) to rank {self._next_server}.")
        self.conduit.send(cycle_data, self.server_ranks[self._next_server])
        self._next_server = (self._next_server + 1) % len(self.server_ranks)

        return False  # Not yet done.

    def process_step(self, i: int, step: AdaLEDStep):
        """Record and print the step, depending on the config."""
        if self.recorder.config.every > 0:
            self.recorder.add_step(i, step)

        log_every = self.client.config.log_every
        if log_every > 0 and (i + 1) % log_every == 0:
            print(i + 1, step, flush=(i + 1) % (10 * log_every) == 0)

    def _is_done(self, i: int) -> bool:
        config = self.client.config
        if self.conduit.shutdown_by_remote:
            if not config.quiet:
                print(f"Server sent shut down request, stopping the client.", flush=True)
            return True
        elif config.max_steps > 0 and i >= config.max_steps:
            if not config.quiet:
                print(f"Reached max steps of {config.max_steps}, stopping.", flush=True)
            return True
        else:
            return False

    def close(self):
        assert self.conduit.active()
        if self.client.active_client_comm:
            # Wait first till all ranks have completed or received a shutdown
            # request from the servers.
            cpu_friendly_barrier(self.client.active_client_comm)
        self.conduit.wait_and_close()
        # Process all messages, as they might include a restart save command.
        for message in self.conduit.pop_all():
            self.client.process_server_data(message.data)
        if self.client._save_restart_to:
            self.save(self.client._save_restart_to)
        self.generator.micro.close()
