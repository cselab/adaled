from adaled.backends import TensorCollection, cmap
from adaled.conduits.base import Conduit, Payload
from adaled.led.diagnostics import AdaLEDCycleDiagnostics, AdaLEDDiagnostics
from adaled.led.misc import init_output_folder
from adaled.led.trainers import LEDTrainer, LEDTrainingOutput
from adaled.led.topology import Topology
from adaled.solvers.base import MicroSolver, MacroSolver
from adaled.transformers.base import Transformer, IdentityTransformer
from adaled.utils.data.collections import DynamicTrajectoryDatasetCollection
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass, field
from adaled.utils.io_ import DumpConfig
from adaled.utils.mpi import cpu_friendly_barrier, mock_world
from adaled.utils.misc import to_numpy
import adaled.utils.io_ as io_

import numpy as np
import torch

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import copy
import dataclasses
import os
import threading
import time
import warnings

nan64 = np.float64(np.nan)

__all__ = ['nan64', 'AdaLEDServerConfig', 'AdaLEDServer']

@dataclass
class AdaLEDServerConfig(DataclassMixin):
    # Maximum number of partial epochs to perform with no new data inbetween.
    max_trains_without_new_data: int = 5

    # Maximum time to wait for new data, after which to continue training.
    max_wait_seconds_for_new_data: float = 1.0

    # How often to compute the validation error, in the number of local cycles.
    validation_every: int = 20

    # Continue from the latest dump files, if any is found.
    restart: bool = False

    # Should restart files be saved?
    save_restart_files: bool = True

    # Format of the restart folder name. restart_count is automatically
    # determined as the 1 + max number in file (folder) names matching this
    # format, or 0 if such files/folders do not exist.
    restart_folder_fmt: str = 'restart-{restart_count:03d}'

    # Learning rate at which to terminate the run. 0 to disable this criterion.
    min_lr: float = 0.0

    # For runs with client.always_run_micro == True, sleep for the
    # corresponding amount of time spent advancing the micro solver during the
    # macro-only stage. Useful to ensure that debug and production runs have
    # similar accuracy.
    compensate_for_stats_overhead: bool = True

    # Reduce verbosity.
    quiet: bool = False

    # Whether to create plot.sh and similar.
    init_output_folder: bool = True

    # Maximum runtime in seconds. If the runtime exceeds the specified
    # threshold, servers will notify the clients to save their state and close.
    # Clients will first finish and send back the current cycle they are
    # running.
    max_seconds: int = 0

    # Maximum number of timesteps (across all processed cycles globally) until
    # which the training is active. Used for testing how disabling training
    # affects the results. 0 to disable.
    max_global_timesteps_for_training: int = 0

    # Following arguments are passed to .format():
    #   cycle: current cycle index
    #   frame: the index of the dump (frame == cycle // every)
    #   rank: (for diagnostics only) current rank of the server communicator
    dump_dataset: DumpConfig = field(
            lambda: DumpConfig(keep_last=0, hardlink=True))
    dump_macro: DumpConfig = field(DumpConfig)
    dump_trainers: DumpConfig = field(DumpConfig)
    dump_transformer: DumpConfig = field(DumpConfig)
    dump_diagnostics: DumpConfig = field(
            # Diagnostics contains all information from the start of the
            # simulation, so no need for latest.
            lambda: DumpConfig(every=10, keep_last=-1, latest_symlink_path=None))

    def __post_init__(self):
        """Fill dump configs with default paths."""
        if self.dump_dataset.path_fmt is SPECIFIED_LATER:
            self.dump_dataset.path_fmt = 'dataset-{frame:05d}/'
        if self.dump_macro.path_fmt is SPECIFIED_LATER:
            self.dump_macro.path_fmt = 'macro-{frame:05d}.pt'
        if self.dump_trainers.path_fmt is SPECIFIED_LATER:
            self.dump_trainers.path_fmt = 'trainers-{frame:05d}.pt'
        if self.dump_transformer.path_fmt is SPECIFIED_LATER:
            self.dump_transformer.path_fmt = 'transformer-{frame:05d}.pt'
        if self.dump_diagnostics.path_fmt is SPECIFIED_LATER:
            self.dump_diagnostics.path_fmt = 'diagnostics-{rank:03d}.pt'


def _check_lr_stopping_criterion(min_lr: float, server: 'AdaLEDServer') -> bool:
    """Check if the min_lr stopping criterion is met.

    Looks for all `lr` keys in `server.trainer.get_hyperparams()`, recursively.
    The stopping criterion is met if the maximum current learning rate is
    approximately smaller than or equal to `min_lr`.
    """
    if not server.trainer:
        return False

    def get_lrs(d: dict):
        for k, v in d.items():
            if k == 'lr':
                yield from np.asarray(v, dtype=np.float64)
            elif isinstance(v, dict):
                yield from get_lrs(v)

    lrs = max(get_lrs(server.trainer.get_hyperparams()))
    return lrs < min_lr * (1 + 1e-6)


def _find_latest_and_next_restart_path(fmt: str) -> Tuple[Optional[str], str]:
    """Search for all files that match the `fmt` format string, find the
    latest one and determine the next restart path.

    In multirank simulations, this
    function should be run collectively (or only on master rank), to ensure
    there is no race condition between the lookup and mkdirs.
    """
    import glob
    import re
    re_id = re.compile(r'\d+')
    max_index = -1
    latest = None
    pattern = re.sub(r'{[^}]*}', '*', fmt)
    for path in glob.glob(pattern):
        match = re_id.findall(os.path.basename(path))
        if match:
            index = max(int(number) for number in match)
            if index > max_index:
                max_index = index
                latest = path
    next_ = fmt.format(restart_count=max_index + 1)
    return (latest, next_)


class AdaLEDServer:
    """Adaptive LED server class, responsible for the dataset and training.

    Attributes:
        comm: (comm) server-side communicator, used when each rank runs both
            server and client alternatively (prefer the `AdaLED` class instead)
        topology: the topology of servers and ranks for multirank runs
    """
    def __init__(
            self,
            config: AdaLEDServerConfig,
            macro: MacroSolver,
            datasets: DynamicTrajectoryDatasetCollection,
            trainer: Optional[LEDTrainer] = None,
            transformer: Transformer = IdentityTransformer(),
            *,
            topology: Optional[Topology] = None,
            comm: Optional['mpi4py.MPI.Intracomm'] = None):
        if topology is not None and comm is not None:
            raise TypeError("cannot specify both `comm` and `topology`")
        if not comm:
            comm = (topology and topology.active_side_comm) or mock_world
        config.validate()
        self.macro = macro
        self.trainer = trainer
        self.transformer = transformer
        self.config = config
        self.datasets = datasets
        self.comm = comm
        self.topology = topology

        self._num_train_rounds = 0
        self._num_global_cycles = 0  # Total # of cycles on all server ranks.
        self.diagnostics = AdaLEDDiagnostics()
        self._last_losses: Optional[TensorCollection] = None
        self._seconds_to_sleep = 0.0

        zero = np.int64(0)
        self._total_train_samples = LEDTrainingOutput(zero, zero, zero, zero)

        # Total number of time steps received from all clients.
        self._total_client_timesteps = 0

        latest_restart_dir, self._save_restart_to = \
                _find_latest_and_next_restart_path(
                        os.path.abspath(config.restart_folder_fmt))
        if not config.save_restart_files:
            self._save_restart_to = None

        self._first_message = {
            'restart_from': latest_restart_dir if config.restart else None,
        }

        if config.restart and latest_restart_dir:
            self.restart(latest_restart_dir)
        elif comm.size > 1:
            # Synchronize initial weights of the macro solver.
            if comm.rank == 0:
                comm.bcast(self.macro.state_dict())
            else:
                self.macro.load_state_dict(comm.bcast(None))

    def state_dict(self) -> dict:
        return {
            'num_train_rounds': self._num_train_rounds,
            'num_global_cycles': self._num_global_cycles,
            'last_losses': self._last_losses,
            'total_train_samples': self._total_train_samples,
            'total_client_timesteps': self._total_client_timesteps,
            'diagnostics': self.diagnostics.state_dict(),
            'macro': self.macro.state_dict(),
            'trainer': self.trainer.state_dict() if self.trainer else None,
            'transformer': self.transformer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._num_train_rounds = state['num_train_rounds']
        self._num_global_cycles = state['num_global_cycles']
        self._last_losses = state['last_losses']
        self._total_train_samples = state['total_train_samples']
        self._total_client_timesteps = state['total_client_timesteps']

        # Optional, in case someone wants to manipulate with state files.
        diagnostics = state.get('diagnostics')
        macro = state.get('macro')
        trainer = state.get('trainer')
        transformer = state.get('transformer')
        if diagnostics:
            self.diagnostics.load_state_dict(diagnostics)
        if macro:
            self.macro.load_state_dict(macro)
        if trainer and self.trainer:
            self.trainer.load_state_dict(trainer)
        if transformer:
            self.transformer.load_state_dict(transformer)

    def save(self, dir: str):
        if self.comm.rank == 0:
            os.makedirs(dir, exist_ok=True)
            io_.save(self.state_dict(), os.path.join(dir, 'server.pt'))
        self.datasets.save(os.path.join(dir, 'dataset'))

    def restart(self, dir: str):
        self.load_state_dict(io_.load(os.path.join(dir, 'server.pt')))
        self.datasets.close()
        self.datasets = self.datasets.load(os.path.join(dir, 'dataset'))

    def prepare_data_for_clients(self, extra: Optional[dict] = None) -> Payload:
        """Return state dicts of the macro solver and the transformer."""
        out = {
            'macro': self.macro.state_dict(),
            'transformer': self.transformer.state_dict(),
        }
        if self._first_message or extra:
            if self._first_message:
                out.update(self._first_message)
                self._first_message = None
            if extra:
                out.update(extra)
            # Do not overwrite this message if client's inbox is full.
            return Payload(out, no_delete=True)
        else:
            return Payload(out, no_delete=False)

    def process_client_data(
            self, data: Sequence[Tuple[Any, AdaLEDCycleDiagnostics]]):
        """Process data gathered by one or multiple clients, see
        `AdaLEDGenerator.get_data_for_server`. This operation is collective.

        The current data contains:
            - trajectories (both data and metadata)
            - cycle diagnostics
        """
        if data:
            print(f"Received client data for {len(data)} cycle(s).")
        decorated_trajectories = []
        for trajectory_batch, diagnostics in data:
            # Data element represents one client cycle, i.e. one simulation
            # trajectory batch. In many cases, particularly huge systems, this
            # will be equal to 1.
            decorated = self.decorate_trajectories(trajectory_batch, diagnostics)
            decorated_trajectories.extend(decorated)

        self.add_trajectories(decorated_trajectories)  # Collective operation.
        for trajectories, diagnostics in data:
            self._add_cycle_diagnostics(diagnostics)
        self._check_should_dump(len(data))

    def decorate_trajectories(
            self,
            trajectories: TensorCollection,
            cycle_diagnostics: AdaLEDCycleDiagnostics):
        """Compute and add metadata (cmp_error, cmp_uncertainty and
        latest_loss*).

        Returns the decorated trajectories. The input is left unchanged.
        """
        # We are modifying the structure, make a shallow copy.
        trajectories = trajectories.map(lambda x: x)

        data = trajectories['trajectory']
        num_traj = len(data)
        traj_len = len(data[0])

        def broadcast1(a):
            a = to_numpy(a)
            assert a.shape[0] == num_traj, (a.shape, num_traj)
            return np.moveaxis(np.broadcast_to(a, (traj_len,) + a.shape), 0, 1)

        # def broadcast2(a):
        #     a = to_numpy(a)
        #     return np.broadcast_to(a, (num_traj, traj_len) + a.shape)

        # Store the comparison error and uncertainty in the trajectory dataset,
        # for potential preferential sampling or replacement policy. Not added
        # in the client side to avoid sending the same data multiple times.
        metadata = trajectories['metadata']
        metadata['cmp_error'] = cmap(broadcast1, cycle_diagnostics.cmp_error)
        metadata['cmp_uncertainty'] = \
                cmap(broadcast1, cycle_diagnostics.cmp_uncertainty)
        if self.trainer:
            # Temporarily transfers to GPU (if CUDA is enabled). Transferring
            # GPU arrays to dataset.add_trajectories may lead to GPU
            # out-of-memory in the case the server receives multiple
            # trajectories in one cycle. For small systems where the whole
            # dataset is stored in the GPU memory this represents a minor
            # performance overhead as the data is transferred twice, which is
            # not problematic since the trajectories are small anyway in such
            # cases.
            data = data.map(torch.tensor)
            metadata['latest_loss'] = self.trainer.compute_losses(data).cpu_numpy()
        # metadata['latest_loss_macro'] = cmap(broadcast2, self.macro.metadata)

        return trajectories

    def _add_cycle_diagnostics(self, diagnostics: AdaLEDCycleDiagnostics):
        """Add client-side cycle diagnostics and dump the dataset, macro
        solver, transformer and accumulated diagnostics if needed."""
        self._total_client_timesteps += diagnostics.total_steps
        if self.config.compensate_for_stats_overhead:
            self._seconds_to_sleep += diagnostics.stats_overhead_seconds

        # Every rank stores its own diagnostics, including the training and
        # validation losses. Note that, in the case of distributed training,
        # the trainer internally does compute the global validation loss when
        # updating the scheduler.
        if self._last_losses is not None:
            # Currently we skip the first cycle because `losses` are not yet
            # available then. If this is changed, update `x_cycles` in the
            # diagnostics plotting script.
            hyperparams = self.trainer.get_hyperparams() if self.trainer else {}
            self.diagnostics.append_cycle(
                    diagnostics, losses=self._last_losses,
                    datasets=self.datasets,
                    trainer_hyperparams=hyperparams,
                    **self.get_custom_cycle_stats())

    def _check_should_dump(self, num_local_new_cycles: int):
        config = self.config
        local_ = np.array([num_local_new_cycles])
        global_ = local_.copy()
        self.comm.Allreduce(local_, global_)

        old = self._num_global_cycles
        self._num_global_cycles += global_[0]
        if global_[0]:
            print(f"Global cycle: {self._num_global_cycles}  "
                  f"(local={num_local_new_cycles})")

        # The for loop is the safest way to ensure we don't accidentally miss
        # some invocation of dump(), especially in a multirank environment,
        # where each rank has its own number of (new) local cycles. Note that
        # the dump frequency should be adjusted to the number of ranks.
        for cycle in range(old, self._num_global_cycles):
            def _dump_dataset(path, **kwargs):
                self.datasets.save(path, self.comm, **kwargs)

            config.dump_dataset.check(
                    cycle=cycle, comm=self.comm, collective=True,
                    dump_func=_dump_dataset)

            if self.comm.rank == 0:
                def _dump_macro(path, **kwargs):
                    io_.save(self.macro, path)

                def _dump_transformer(path, **kwargs):
                    io_.save(self.transformer, path)

                def _dump_trainers(path, **kwargs):
                    io_.save(self.trainer.state_dict(), path)

                config.dump_macro.check(cycle=cycle, dump_func=_dump_macro)
                config.dump_transformer.check(cycle=cycle, dump_func=_dump_transformer)
                config.dump_trainers.check(cycle=cycle, dump_func=_dump_trainers)

            config.dump_diagnostics.check(
                    cycle=cycle, dump_func=self.diagnostics.save,
                    comm=self.comm, collective=False)

    def get_custom_cycle_stats(self) -> Dict[str, Any]:
        """Customization of stats."""
        return {
            'total_train_samples': dataclasses.asdict(self._total_train_samples),
        }

    def add_trajectories(self, decorated_trajectories: Sequence[TensorCollection]):
        self.datasets.add_trajectories(decorated_trajectories)
        self.diagnostics.train_dataset_size = \
                len(self.datasets.train_dataset.as_trajectories())
        self.diagnostics.valid_dataset_size = \
                len(self.datasets.valid_dataset.as_trajectories())

    def train(self):
        self._last_losses = self._train()

    def _train(self) -> TensorCollection:
        """Retrain the RNNs. Return a collection of losses."""
        if not self.trainer:
            return TensorCollection()
        version = self.macro.metadata['version']
        should_validate = (version + 1) % self.config.validation_every == 0
        if should_validate:
            valid_dataset = self.datasets.valid_dataset
        else:
            valid_dataset = None

        kwargs = {'total_client_timesteps': self._total_client_timesteps}
        losses, indices = self.trainer.train(
                self.datasets.train_dataset, valid_dataset,
                sample_count_policy_kwargs=kwargs)
        self.macro.metadata['version'] = version + 1
        print("Macro version:", version, flush=True)
        self._update_latest_losses(
                losses, indices, self.datasets.train_dataset, valid_dataset)
        return self._postprocess_losses(losses)

    def _update_latest_losses(
            self, losses: LEDTrainingOutput, indices: LEDTrainingOutput,
            train_dataset, valid_dataset):
        """Update dataset latest losses and update _total_train_samples."""
        # The idea here was to store the metadata (version) of the macro solver
        # used the latest the state / trajectory was used for training.
        # However, it seems like we would need two different numbers, one for
        # latest transformer, one for latest macro training, since currently
        # state sampling for the transformer is decoupled from trajectory
        # sampling for the macro. Such stats seem overly complicated. If
        # needed, they can be enabled.
        # macro_metadata = self.macro.metadata

        def _remove_adversarial(losses):
            """Remove the adversarial part of the loss, to make it consistent
            with `compute_losses`. Move to CPU."""
            # FIXME: This is hacky, there should be a generic way of doing this.
            if isinstance(losses, TensorCollection):
                return losses.get('original', losses).cpu()
            else:
                return to_numpy(losses)

        def _process1(dataset, attr):
            losses_ = getattr(losses, attr)
            if losses_ is not None and dataset is not None:
                dataset.update_states(
                        getattr(indices, attr), ('metadata', 'latest_loss', 'transformer'),
                        _remove_adversarial(losses_))  # Hack.
                # dataset.update_states(
                #         indices, ('metadata', 'latest_loss_macro'), macro_metadata)
                setattr(stats, attr, getattr(stats, attr) + len(losses_))

        def _process2(dataset, attr):
            losses_ = getattr(losses, attr)
            if losses_ is not None and dataset is not None:
                dataset.update_trajectories(
                        getattr(indices, attr), ('metadata', 'latest_loss', 'macro'),
                        _remove_adversarial(losses_[:, np.newaxis]))  # Hack.
                # dataset.update_trajectories(
                #         indices, ('metadata', 'latest_loss_macro'), macro_metadata)
                setattr(stats, attr, getattr(stats, attr) + len(losses_))

        stats = self._total_train_samples
        _process1(train_dataset, 'transformer_train')
        _process1(valid_dataset, 'transformer_valid')
        _process2(train_dataset, 'macro_train')
        _process2(valid_dataset, 'macro_valid')

    def _postprocess_losses(self, losses: LEDTrainingOutput):
        """Replace None losses with NaNs, compute mean loss (mean of all
        states/trajectories of the partial epoch) and put into a
        TensorCollection."""
        train1 = losses.transformer_train
        valid1 = losses.transformer_valid
        train1 = train1.mean() if train1 is not None else nan64
        valid1 = valid1.mean() if valid1 is not None else nan64 * train1

        train2 = losses.macro_train.mean()
        valid2 = losses.macro_valid
        valid2 = valid2.mean() if valid2 is not None else nan64 * train2
        return TensorCollection(
                transformer_train=train1,
                transformer_valid=valid1,
                macro_train=train2,
                macro_valid=valid2)

    def run(self, *args, **kwargs):
        runner = self.make_runner(*args, **kwargs)
        runner.run()

    def make_runner(self, **kwargs):
        topology = self.topology
        if topology:
            return _ServerRunner(self, topology.active_intercomm,
                                 topology.remote_activecomm_ranks, **kwargs)
        else:
            return _ServerRunner(self, None, [0], **kwargs)


class _ServerRunner:
    def __init__(self,
                 server: AdaLEDServer,
                 intercomm: 'MPI.Intercomm',
                 client_ranks: Sequence[int],
                 conduit: Optional[Conduit] = None):
        self.server = server

        self.latest_data_for_clients = None

        self._stop_local = np.array([0])
        self._stop = np.array([0])
        self._steps_since_last_new_data = 0
        self._start_time = time.time()

        if not conduit:
            if intercomm:
                from adaled.conduits.mpi import IntercommConduit
                conduit = IntercommConduit(intercomm)
            else:
                raise TypeError("conduit must be set manually in non-mpi runs")
        self.conduit = conduit
        self.client_ranks = client_ranks  # With respect to intercomm.

    def __del__(self):
        if self.conduit.active():
            import warnings
            import sys
            warnings.warn("the server runner was not closed")
            sys.stdout.flush()
            sys.stderr.flush()
            self.close()

    def start(self):
        self.conduit.start()
        if self.server.comm.rank == 0 and self.server.config.init_output_folder:
            init_output_folder('.', verbose=not self.server.config.quiet)

    def run(self):
        t0 = time.time()
        self.start()
        try:
            while True:
                if self.step():
                    self.close()
                    break
        except:
            if self.conduit:
                self.conduit.abort()
            raise
        print("Server execution time:", time.time() - t0)

    def step(self, no_sleep: bool = False) -> bool:
        """Exchange data with clients and perform one fractional epoch of
        training. Server-side collective operation.

        Returns if the run stopped. In that case, no further invocations of
        `step()` are allowed. The user must invoke `close()` manually.

        Args:
            no_sleep (bool): disable sleeping (only partially implemented)
        """
        config = self.server.config
        assert self._active()
        waited_for_seconds = 0.0
        if self.client_ranks and self._steps_since_last_new_data >= \
                config.max_trains_without_new_data:
            i = 0
            t0 = time.time()
            while not self.conduit.data_available():
                time.sleep(0.1)
                i += 1
                if self.conduit.shutdown_by_remote:
                    break
                if time.time() - t0 > config.max_wait_seconds_for_new_data:
                    break
            if i > 0:
                waited_for_seconds = time.time() - t0
                print(f"Waited for {waited_for_seconds:.3f}s.")

        client_data = [message.data for message in self.conduit.pop_all()]
        self.server.process_client_data(client_data)  # <-- collective operation

        # Waiting should happen only at the beginning of the simulation (when
        # macro is not accepted), so this does not have a large effect, but it
        # is semantically more correct (to emulate how the simulation would
        # look like without always_run_micro=1).
        to_sleep = self.server._seconds_to_sleep - waited_for_seconds
        self.server._seconds_to_sleep = 0
        if not no_sleep and to_sleep > 0:
            print(f"Sleeping for {to_sleep}s.")
            time.sleep(to_sleep)

        comm = self.server.comm
        if (comm and comm.size > 1 and comm.allreduce(len(client_data)) > 0) \
                or len(client_data) > 0:
            self._steps_since_last_new_data = 0

        _curr_steps = self.server._total_client_timesteps
        _disable_after = config.max_global_timesteps_for_training
        if _disable_after == 0 or _curr_steps <= _disable_after:
            self.server.train()
        else:
            print(f"Client timesteps {_curr_steps} exceeded limit "
                  f"{_disable_after}, training disabled.")

        done = self._is_done()
        if done and config.save_restart_files:
            extra = {'save_restart_to': self.server._save_restart_to}
        else:
            extra = {}

        self.latest_data_for_clients = \
                copy.deepcopy(self.server.prepare_data_for_clients(extra))
        for client_rank in self.client_ranks:
            self.conduit.send(self.latest_data_for_clients, client_rank)

        self._steps_since_last_new_data += 1
        return done

    def _is_done(self) -> bool:
        config = self.server.config
        if self.conduit.shutdown_by_remote:
            if not config.quiet:
                print(f"Client sent stop request, stopping the server.", flush=True)
            stop = True
        elif config.min_lr > 0 and \
                _check_lr_stopping_criterion(config.min_lr, self.server):
            if not config.quiet:
                print(f"Reached min learning rate of {config.min_lr}, stopping.", flush=True)
            stop = True
        elif config.max_seconds and time.time() - self._start_time > config.max_seconds:
            if not config.quiet:
                print(f"Reached max runtime of {config.max_seconds}s, stopping.", flush=True)
            stop = True
        else:
            stop = False

        self._stop_local[0] = stop
        if self.server.comm.size > 1:
            from mpi4py import MPI
            self.server.comm.Allreduce(self._stop_local, self._stop, MPI.MAX)
        else:
            self._stop[:] = self._stop_local
        stop = bool(self._stop[0])
        return stop

    def _active(self) -> bool:
        return self.conduit.active()

    def close(self):
        """Close connection to clients. Server-side collective operation.

        Note: if the intercomm is not provided, the client must be shut down
        manually (see non-distributed LED).

        After invoking `close()`, no further invocations of `step()` are
        allowed.
        """
        assert self._active()
        comm = self.server.comm
        # As opposed to the clients, no need for a barrier here at the server
        # side, since they are synchronized anyway (see `_if_done`).
        self.conduit.wait_and_close()

        # Process all remaining messages, in case there are some trajectories
        # left and we want a restart file.
        client_data = [message.data for message in self.conduit.pop_all()]
        self.server.process_client_data(client_data)  # <-- collective operation

        if self.server._save_restart_to and self.server.config.save_restart_files:
            self.server.save(self.server._save_restart_to)
