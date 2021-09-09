from .config import MicroConfigBase
from .utils_2d import compute_vorticity_2d_no_boundary, stream_function_to_velocity
from .utils_3d import curl
from adaled import TensorCollection, to_numpy
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass, field
from adaled.transformers.multiresolution import MRLayerConfig
import adaled

import numpy as np
import torch

from enum import Enum
from typing import List, Optional, Tuple, Union, Sequence, TYPE_CHECKING
import copy
import math
import signal
import sys
import time

if TYPE_CHECKING:
    from mpi4py import MPI

_Array = Union[np.ndarray, 'torch.Tensor']

class CUPTensorCollection(TensorCollection):
    """Tensor collection extended with hidden data not stored in the dataset.
    This is a somewhat hacky approach to pass extra information from
    get_state() to the comparison error function.

    Attributes:
        full_resolution_state: fluid field array (velocity + optionally
                vorticity + optionally pressure), before downsampling
    """
    __slots__ = ('full_resolution_state',)

    full_resolution_state: np.ndarray


class MicroStateType(Enum):
    # State as returned by get_state: (velocity, optional vorticity, optional pressure)
    GET_STATE = 1

    # State as returned by the decoder: (potential or velocity, optional pressure)
    DECODER = 2

    # # State as expected by the update_state_impl: (velocity, optional pressure)
    # IMPORT = 3


class MicroStateHelper:
    """Helper functions for extracting specific fields from a micro state,
    which is for performance reasons kept as a single concatenated array."""
    def __init__(self, config: MicroConfigBase):
        self.state_has_vorticity = config.vorticity_in_state
        self.state_has_pressure = config.pressure_in_state
        self.hs = config.compute_hs()
        self.ndim = len(config.cells)
        assert 2 <= self.ndim <= 3

    def layer_to_velocity(self, channels: _Array) -> _Array:
        return channels[:, :self.ndim, ...]

    def layer_to_vorticity_no_boundary(self, layer: int, channels: _Array) -> _Array:
        if self.state_has_vorticity:
            if self.ndim == 2:
                return channels[:, 2, 1:-1, 1:-1]
            elif self.ndim == 3:
                return channels[:, 3:6, 1:-1, 1:-1, 1:-1]
        else:
            if self.ndim == 2:
                return compute_vorticity_2d_no_boundary(channels, self.hs[layer])
            elif self.ndim == 3:
                return curl(channels, self.hs[layer])
        raise NotImplementedError("unreachable")

    def layer_to_vorticity_with_boundary(self, layer: int, channels: np.ndarray) -> np.ndarray:
        vort = self.layer_to_vorticity_no_boundary(layer, channels)
        if self.ndim == 2:
            assert vort.ndim == 3, (vort.shape, channels.shape)
            vort = np.pad(vort, ((0, 0), (1, 1), (1, 1)))
        else:
            assert vort.ndim == 5, (vort.shape, channels.shape)
            vort = np.pad(vort, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)))
        return vort

    def layer_to_pressure(self, channels: _Array) -> Optional[_Array]:
        if self.state_has_pressure:
            return channels[:, -1, ...]
        else:
            return None

    def layer_to_collection(
            self, layer: int, channels: _Array, no_compute: bool = True) -> TensorCollection:
        out = {'v': self.layer_to_velocity(channels)}
        if self.state_has_vorticity or not no_compute:
            out['vort'] = self.layer_to_vorticity_no_boundary(layer, v)
        if self.state_has_pressure:
            out['p'] = self.layer_to_pressure(channels)
        return TensorCollection(out)

    def layer_to_import_layer(self, layer: int, channels: _Array) -> np.ndarray:
        v = self.layer_to_velocity(channels)
        p = self.layer_to_pressure(channels)
        return v if p is None else np.concatenate([v, p[:, None]], axis=1)

    def layer_to_v_vort_p(self, layer: int, channels: np.ndarray) -> np.ndarray:
        """For visualization."""
        v = self.layer_to_velocity(channels)
        vort = self.layer_to_vorticity_with_boundary(layer, channels)
        p = self.layer_to_pressure(channels)
        if self.ndim == 2:
            vort = vort[:, np.newaxis]
        parts = [v, vort, p[:, np.newaxis]] if p is not None else [v, vort]
        return np.concatenate(parts, axis=1)


class AutoencoderReconstructionHelper:
    """Reconstruct velocity and other fields from the neural network output,
    which is either the velocity itself (in case nothing has to be done) or the
    stream / potential function, plus optionally the pressure."""
    def __init__(self, config: MicroConfigBase, upscaled: bool = False):
        """
        Arguments:
            upscaled: (bool) Set to True if queries will be performed on the
                    upscaled and merged layers, not on individual layers. In
                    that case, `layer` indices must be 0.
        """
        super().__init__()
        self.predict_potential_function = config.predict_potential_function
        self.predict_pressure = config.predict_pressure
        self.ndim = len(config.cells)

        if upscaled:
            self.hs = [config.compute_h()]
        else:
            self.hs = config.compute_hs()

    def layer_to_stream(self, channels: _Array) -> Optional[_Array]:
        if not self.predict_potential_function:
            return None
        if self.ndim == 2:
            return channels[:, 0, :, :]
        elif self.ndim == 3:
            return channels[:, :3, :, :, :]
        else:
            raise NotImplementedError(self.ndim)

    def layer_to_velocity(self, layer: int, channels: _Array) -> _Array:
        if not self.predict_potential_function:
            return channels[:, :self.ndim, ...]  # Already v.
        if self.ndim == 2:
            return stream_function_to_velocity(channels[:, :1, :, :], self.hs[layer])
        elif self.ndim == 3:
            return curl(channels[:, :3, :, :, :], self.hs[layer])
        else:
            raise NotImplementedError(self.ndim)

    def v_to_vorticity_no_boundary(self, layer: int, v: _Array) -> _Array:
        """Compute vorticity (without the boundary) from the velocity field."""
        if self.ndim == 2:
            return compute_vorticity_2d_no_boundary(v, self.hs[layer])
        elif self.ndim == 3:
            return curl(v, self.hs[layer])
        else:
            raise NotImplementedError(self.ndim)

    def v_to_vorticity_with_boundary(self, layer: int, v: np.ndarray) -> np.ndarray:
        vort = self.v_to_vorticity_no_boundary(layer, v)
        if self.ndim == 2:
            vort = np.pad(vort, ((0, 0), (1, 1), (1, 1)))
        else:
            vort = np.pad(vort, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)))
        return vort

    def layer_to_pressure(self, channels: _Array) -> Optional[_Array]:
        if self.predict_pressure:
            if self.predict_potential_function:
                # Potential function has padding of 1 on all sides. Ignore the
                # padding of the pressure, it has no meaning.
                if self.ndim == 2:
                    return channels[:, -1, 1:-1, 1:-1]
                else:
                    return channels[:, -1, 1:-1, 1:-1, 1:-1]
            return channels[:, -1, ...]
        else:
            return None

    def layer_to_import_layer(self, layer: int, channels: _Array) -> np.ndarray:
        v = to_numpy(self.layer_to_velocity(layer, channels))
        p = self.layer_to_pressure(channels)
        if p is not None:
            p = to_numpy(p)
            return np.concatenate([v, p[:, None]], axis=1)
        else:
            return v

    def layer_to_collection(self, layer: int, channels: _Array) -> TensorCollection:
        v = self.layer_to_velocity(layer, channels)
        vort = self.v_to_vorticity_no_boundary(layer, v)
        out = {'v': v, 'vort': vort}
        if self.predict_pressure:
            out['p'] = self.layer_to_pressure(channels)
        return TensorCollection(out)

    def layer_to_v_vort_p(self, layer: int, channels: np.ndarray) -> np.ndarray:
        """For visualization."""
        v = self.layer_to_velocity(layer, channels)
        vort = self.v_to_vorticity_with_boundary(layer, v)
        p = self.layer_to_pressure(channels)
        if self.ndim == 2:
            vort = vort[:, np.newaxis]
        parts = [v, vort, p[:, np.newaxis]] if p is not None else [v, vort]
        return np.concatenate(parts, axis=1)


class AutoencoderReconstructionLayer(AutoencoderReconstructionHelper, torch.nn.Module):
    LAYER_SHORT_NAME = "Rec"  # Short name for histogram plots.

    def forward(self, x: TensorCollection) -> TensorCollection:
        def expand(multikey, array):
            if multikey[0] == 'layers':
                k = int(multikey[-1][5:])  # f'layer{k}'
                return self.layer_to_collection(k, array)
            else:
                return array

        return x.named_map(expand)


def init_dump_signal_handler(sim: 'CUPSolverBase'):
    """Add a signal handler for the SIGUSR1 signal, on which the simulation
    state will be dumped for visualization purposes."""
    def handler(signum, frame):
        if signum == signal.SIGUSR1:
            print(f"Received signal SIGUSR1, dumping Cubism state.")
            # Do not dump immediately, as something might go wrong.
            sim.push_dump('sigusr1')

    signal.signal(signal.SIGUSR1, handler)
    print("Cubism signal handler for SIGUSR1 installed.")


class CUPSolverBase(adaled.MicroSolver):
    """Base micro solver class for CubismUP2D/3D.

    Implements the communication logic between the root rank (adaled-active)
    and non-root ranks (adaled-inactive).
    Handles multiresolution downscaling and upscaling of the velocity field.
    """
    MSG_SIMULATE = 1
    MSG_SIMULATE_AND_GET = 2
    MSG_CLOSE = 3
    MSG_GET_STATE = 4
    MSG_UPDATE_STATE = 5

    def __init__(
            self,
            *,
            config: MicroConfigBase,
            max_v: float,
            argv: List[str] = [],
            comm: Optional['MPI.Intracomm'] = None,
            **kwargs):
        if comm is None:
            global MPI
            from mpi4py import MPI
            comm = MPI.COMM_SELF
        if config.pressure_in_state and not config.predict_pressure:
            raise NotImplementedError(
                    "pressure_in_state==True and predict_pressure==False is "
                    "not supported, because the get_state() invoked "
                    "immediately after update_state() would return an "
                    "outdated pressure. To fix that limitation, CUP2D/3D "
                    "would need to export an API for recomputing the pressure.")

        self.config = config
        self.cells = tuple(config.cells)
        self.comm = comm
        self.multiresolution = config.make_multiresolution()

        self.step = 0
        self._msg_buffer = np.array([0, 0], dtype=np.int32)
        self._last_wall_time = None

        self._dump_queue: List[str] = []
        self._dump_counter = 0
        self.dtype_np = config.get_dtype_np()

        dt_micro, self.micro_per_macro = self.compute_dt_internal(config, max_v)
        self.sim = self.init_simulation(config, dt_micro, comm, **kwargs)

        self.micro_helper = MicroStateHelper(config)
        self.ae_helper = AutoencoderReconstructionHelper(config)
        self.default_update_state_type = MicroStateType.DECODER

        init_dump_signal_handler(self.sim)

        if config.profile:
            from cProfile import Profile
            self.step_profile = Profile()
            self.update_profile = Profile()
        else:
            self.step_profile = None
            self.update_profile = None

        if config.enforce_obstacle_velocity:
            self.enforce_v_mask = config.compute_obstacle_interior_mask()
        else:
            self.enforce_v_mask = None

    def state_dict(self) -> dict:
        # For simplicity, we use get_state(), despite that not being the
        # actual state of the solver considering AMR etc.
        return {
            # Store the whole field, including potential vorticity and
            # pressure. Otherwise, get_state() right after loading the
            # simulation would return garbage pressure.
            # We could improve this by first converting to the import
            # (internal) state, but it's not that important.
            'state': self.get_state(),
            'step': self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = 0
        if self.comm.rank == 0:
            self.update_state(state['state'], skip_steps=state['step'],
                              state_type=MicroStateType.GET_STATE)

    @property
    def num_import_channels(self) -> int:
        ndim = len(self.cells)
        return ndim + (1 if self.config.pressure_in_state else 0)

    @property
    def import_state_shape(self) -> int:
        return (1, self.num_import_channels, *self.cells[::-1])

    def init_simulation(
            self,
            config: MicroConfigBase,
            dt_micro: float,
            comm: 'MPI.Intracomm',
            **kwargs) -> Union['cup2d.Simulation', 'cup3d.Simulation']:
        """Init the simulation, operators and obstacles. Return the simulation object."""
        raise NotImplementedError()

    def advance_impl(self, cup_steps: int) -> None:
        """Advance the state by `cup_steps` internal steps."""
        raise NotImplementedError(self)

    def dump_state(self, prefix: str) -> None:
        """Dump state for debugging and visualization purposes."""
        raise NotImplementedError(self)

    def get_local_state(self) -> np.ndarray:
        """Return the state as a numpy array of shape (1, num_channels, [z,] y, x)."""
        raise NotImplementedError(self)

    def get_mock_state(self) -> TensorCollection:
        """Return a mock state of shape (1, num_channels, [z,] y, x)."""
        raise NotImplementedError(self)

    def get_quantities_of_interest(self, for_state: bool) -> Union[dict, TensorCollection]:
        """Return macroscopic values of interest. Empty by default. All
        returned values must be numpy arrays or numpy scalars.

        If `for_state == True`, return only the data that the macro propagator
        must be able to reproduce. The case `for_state == False` is for
        recording and may include additional information.

        Arguments:
            for_state: (bool) True if for get_state, False if for recording
        """
        # TODO: Don't complicate with for_state=False/True, make
        # get_quantities_of_interest always return for_state=True, and then add
        # another record for extra qoi.
        return {}

    def update_state_impl(self, merged: np.ndarray, skip_steps: int) -> None:
        """Update the velocity field and optionally pressure, update .step and
        the internal current time."""
        assert merged.shape == self.import_state_shape
        raise NotImplementedError(self)

    def push_dump(self, prefix: str):
        self._dump_queue.append(f'{prefix}_{self._dump_counter:05d}_')
        self._dump_counter += 1

    @staticmethod
    def compute_dt_internal(config: MicroConfigBase, max_v: float, verbose: bool = True) \
            -> Tuple[float, int]:
        """Compute dt_internal and the number of internal steps per macro step."""
        min_v = max_v / 2  # TODO
        h = config.extent / np.max(config.cells)
        dt_advection = config.cfl * h / max_v
        dt_diffusion = 0.25 * h * h / (config.nu + 0.25 * h * min_v)
        dt_internal = min(dt_advection, dt_diffusion)  # internal
        internal_per_macro = int(np.ceil(config.dt_macro / dt_internal))
        rounded_dt_internal = config.dt_macro / internal_per_macro
        final_cfl = rounded_dt_internal / h * max_v
        if verbose:
            print(f"Cubism time step for min_v={min_v!r} and max_v={max_v!r}:\n"
                  f"    dt_advection=cfl*h/max_v={dt_advection!r} (including CFL)\n"
                  f"    dt_diffusion=h^2/(nu+h*min_v/4)/4={dt_diffusion!r}\n"
                  f"    dt_internal={dt_internal!r}\n"
                  f"    rounded_dt_internal={rounded_dt_internal!r}\n"
                  f"    internal_per_macro={internal_per_macro}\n"
                  f"    cells={config.cells}\n"
                  f"    extent={config.extent}\n"
                  f"    target_CFL={config.cfl}\n"
                  f"    final_CFL={final_cfl}\n"
                  f"    max_v={max_v}\n", end="", flush=True)
        return rounded_dt_internal, internal_per_macro

    def close(self):
        """In multirank simulations, inform non-root ranks to stop the simulation."""
        assert self.sim is not None, "close() already invoked"
        assert self.comm.rank == 0, "may be run only from the simulation root rank"
        if self.comm.size > 1:
            print("Closing Cubism, informing other simulation ranks.", flush=True)
            self._send_message_to_non_root(self.MSG_CLOSE)
        self.sim = None

    def close_non_root(self):
        assert self.sim is not None
        assert self.comm.rank > 0, "may be run only from non-root simulation ranks"
        self.sim = None

    def _send_message_to_non_root(self, msg: int, param: int = 0):
        """Send message to non-root ranks."""
        assert self.comm.rank == 0
        buf = self._msg_buffer
        buf[0] = msg
        buf[1] = param
        self.comm.Bcast(buf)

    def _recv_message_from_root(self):
        """Receive message from the root rank."""
        assert self.comm.rank > 0
        self.comm.Bcast(self._msg_buffer)
        return self._msg_buffer

    def run_non_root(self):
        """Execute requests from the root rank until a close signal is
        received."""
        assert self.comm.rank > 0, \
               "this function may be invoked only from non-root Cubism ranks"
        state_buffer = np.empty((1, self.num_import_channels, *self.cells[::-1]))
        while True:
            msg, param = self._recv_message_from_root()
            if msg == self.MSG_CLOSE:
                print("Received close message from the root rank, breaking.", flush=True)
                self.close_non_root()
                break
            elif msg == self.MSG_SIMULATE_AND_GET or msg == self.MSG_GET_STATE:
                if msg == self.MSG_SIMULATE_AND_GET:
                    self._advance(param)

                # get_state() is invoked at the end of advance().
                out = self.get_local_state()
                self.comm.Reduce(out, None)
            elif msg == self.MSG_SIMULATE:
                self._advance(param)
            elif msg == self.MSG_UPDATE_STATE:
                self.comm.Bcast(state_buffer)
                self.update_state_impl(state_buffer, skip_steps=param)
            else:
                raise ValueError(msg)

    def advance(self, F: np.ndarray, no_adaled: bool = False) \
            -> Optional[CUPTensorCollection]:
        if self.step_profile:
            self.step_profile.enable()
        if self.comm.size > 1 and not self.config.mock:
            self._send_message_to_non_root(
                    self.MSG_SIMULATE if no_adaled else self.MSG_SIMULATE_AND_GET, k)
        self._advance(F)

        if self._dump_queue:
            self.dump_state(self._dump_queue.pop())

        # Non-root ranks assume that advance() is always followed by a
        # get_state(), so don't send an additional message.
        out = None if no_adaled else self.get_state(_send_msg=False)
        if self.step_profile:
            self.step_profile.disable()
            if self.step % 30 == 0:
                self.step_profile.print_stats(sort='tottime')
                print(flush=True)
        return out

    def _advance(self, F: np.ndarray):
        if self.step == 0 and not self.config.mock:
            self._rampup()

        t0 = time.time()
        if self.config.mock:
            # Quickly pass through first 1000 steps, then slow down to reduce
            # the record file sizes.
            time.sleep(0.01 if self.step < 1000 else 0.2)
        else:
            self.advance_impl(self.micro_per_macro)
        self.step += 1
        t1 = time.time()
        wall_time = t1 - t0

        # Total time between two consecutive launches of _advance().
        global_wall_time = t1 - self._last_wall_time if self._last_wall_time else np.nan

        self._last_wall_time = t1
        if self.comm.rank == 0:
            self.log_step(wall_time, global_wall_time)

    def _rampup(self):
        for i in range(self.config.rampup_steps):
            t0 = time.time()
            self.advance_impl(1)
            t1 = time.time()
            if self.comm.rank == 0:
                num_blocks = len(self.sim.fields.chi.blocks)
                print(f"RAMPUP #{i}/{self.config.rampup_steps}: "
                      f"{t1 - t0:6.3f}s  {num_blocks:5} blocks\n",
                      end="", file=sys.stderr, flush=True)

    def log_step(self, wall_time: float, global_wall_time: float):
        num_blocks = len(self.sim.fields.chi.blocks)
        ms_per_block = 1000 * wall_time / num_blocks
        print(f"STEP #{self.step}: {wall_time:6.3f}s  {global_wall_time:6.3f}s"
              f"  {num_blocks:5} blocks  {ms_per_block:5.1f}ms/block\n",
              end="", file=sys.stderr, flush=True)

    def get_state(self, _send_msg: bool = True) -> CUPTensorCollection:
        if self.config.mock:
            out = self.get_mock_state()
            out.dict.update(self.get_quantities_of_interest(for_state=True))
            return out
        if self.comm.size > 1:
            # get_local_state is a collective operation due to stencil-based
            # interpolation, inform the worker ranks first.
            if _send_msg:
                self._send_message_to_non_root(self.MSG_GET_STATE)
            state = self.get_local_state()  # (1, 2 or 3, [z,] y, x)
            global MPI
            from mpi4py import MPI
            self.comm.Reduce(MPI.IN_PLACE, state)
        else:
            state = self.get_local_state()  # (1, 2 or 3, [z,] y, x)

        layers = self._downscale(state)

        for key, value in layers.allitems():
            assert value.dtype == self.dtype_np, (key, value.shape, value.dtype)

        # Put layers into an additional level of hierarchy, to be able to add
        # other fields (e.g. obstacle states) without interfering with the
        # layers.
        qoi = self.get_quantities_of_interest(for_state=True)
        out = CUPTensorCollection(layers=layers, **qoi)
        out.full_resolution_state = state

        # Uncomment to debug update_state and its effect on the solver performance.
        # if self.step >= 10 and self.step % 30 == 0:
        #     dummy = 1 * out
        #     self.update_state(dummy, skip_steps=0, state_type=MicroStateType.GET_STATE)

        return out

    def _downscale(self, state: np.ndarray) -> TensorCollection:
        parts = self.multiresolution.slice_and_downscale(state)
        return TensorCollection({
            f'layer{i}': part for i, part in enumerate(parts)
        })

    def update_state(
            self,
            new_state: TensorCollection,
            skip_steps: int = 1,
            state_type: Optional[MicroStateType] = None) -> None:
        """Update the state of the CubismUP solver.

        The state can be either the decoder output or the output of
        get_state(), used for the restart mechanism.
        """
        if state_type is None:
            state_type = self.default_update_state_type
        if self.update_profile:
            self.update_profile.enable()

        # Just in case, ensure the correct order by sorting.
        layers = new_state['layers']
        layers = [layers[key] for key in sorted(layers.keys())]

        # The potential function -> (vx, vy[, vz]) conversion has to be
        # done before merging layers, because the potential is defined only
        # up to an unknown constant which makes merging impossible.
        # Convert to CPU before rebuilding, because of memory limitations and
        # because current multiresolution only support CPU reconstruction.
        if state_type == MicroStateType.GET_STATE:
            layers = [self.micro_helper.layer_to_import_layer(i, layer)
                      for i, layer in enumerate(layers)]
        elif state_type == MicroStateType.DECODER:
            layers = [self.ae_helper.layer_to_import_layer(i, layer)
                      for i, layer in enumerate(layers)]
        else:
            raise ValueError(state_type)

        merged = self.multiresolution.rebuild_unsafe(layers)

        if self.config.mock:
            return  # After rebuilding, to check that the shapes are good.
        if self.comm.size > 1:
            self._send_message_to_non_root(self.MSG_UPDATE_STATE, skip_steps)
            self.comm.Bcast(merged)
        self.update_state_impl(merged, skip_steps)

        if self.update_profile:
            self.update_profile.disable()
            self.update_profile.print_stats(sort='tottime')
            print(flush=True)
