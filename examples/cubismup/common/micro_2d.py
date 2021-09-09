from .config import Micro2DConfigBase
from .micro import CUPSolverBase
from .utils_2d import compute_divergence_2d, stream_function_to_velocity
from adaled import TensorCollection
from adaled.utils.dataclasses_ import dataclass

import cubismup2d as cup2d
import libcubismup2d as libcup2d
import numpy as np

from typing import Callable, Optional, Tuple, TYPE_CHECKING
import math
import sys

if TYPE_CHECKING:
    from mpi4py import MPI

class SymmetryBreakingOperator(cup2d.Operator):
    def __init__(self,
                 sim: cup2d.Simulation,
                 obstacle: libcup2d._Shape,
                 uv_func: Callable[[float], Tuple[float, float]],
                 *,
                 dx: float = 0.0,
                 dy: float = 0.0,
                 omega: float,
                 decay: float):
        super().__init__(sim)
        self.uv_func = uv_func
        self.obstacle = obstacle
        # Each rank is expected to have its own seed.
        # FIXME: Should randomization be done be the caller instead?
        random = np.random.uniform(0.8, 1.2, 4)
        self.dx = dx * random[0]
        self.dy = dy * random[1]
        self.omega = omega * random[2]
        self.decay = decay * random[3]
        print(f"Symmetry breaking parameters ({self.__class__.__name__}):\n"
              f"    Reference:  dx={dx}  dy={dy}  omega={omega}  decay={decay}\n"
              f"    Randomized: dx={self.dx}  dy={self.dy}  omega={self.omega}  "
              f"decay={self.decay}  (current rank)")

    def __call__(self, dt: float):
        t = self.data.time
        u, v = self.uv_func(t)

        decay = math.exp(-self.decay * t)
        if decay > 1e-30:
            sin = math.sin(self.omega * t)
            cos = math.cos(self.omega * t)
            tmp = decay * (self.omega * cos - self.decay * sin)
            u += tmp * self.dx
            v += tmp * self.dy

        self.obstacle.u = u
        self.obstacle.v = v


class CUP2DSolver(CUPSolverBase):
    dtype_cup = np.float64

    def init_simulation(self,
                        config: Micro2DConfigBase,
                        dt_micro: float,
                        comm: 'MPI.Intracomm',
                        **kwargs) -> cup2d.Simulation:
        nlevels, cup_cells = config.compute_nlevels_and_cup_cells()
        sim = cup2d.Simulation(
                cells=cup_cells, extent=config.extent, nu=config.nu,
                nlevels=nlevels, start_level=min(nlevels - 1, 1),
                # Pass CFL 0.0 because we manually control the dt.
                cfl=0.0, dt=dt_micro, tdump=0.0,
                verbose=config.verbose, mute_all=config.mute_all,
                output_dir=config.output_dir, comm=comm,
                argv=config.extra_cubism_argv, **kwargs)
        self._data: cup2d.SimulationData = sim.data
        self.add_obstacles_and_operators(sim)
        sim.init()
        return sim

    # virtual
    def add_obstacles_and_operators(self, sim: cup2d.Simulation):
        pass

    def advance_impl(self, cup_steps: int):
        self.sim.simulate(nsteps=cup_steps)

    def dump_state(self, prefix: str):
        self._data.dump_tmp(prefix)
        self._data.dump_vel(prefix)

    def get_local_state(self) -> np.ndarray:
        """Return the state as an ndarray of shape (2 or 3, [grid_z], grid_y, grid_x),
        where the first dimension represents the channels vx, vy and optionally
        vorticity."""
        # Fill with zeros, in order to sum up results from different ranks.
        vel = self.sim.fields.vel.to_uniform(fill=0.0)  # (y, x, 2)
        vel = np.moveaxis(vel, 2, 0)                    # (2, y, x)
        assert vel.shape == (2, self.cells[1], self.cells[0]), \
               (vel.shape, (2, self.cells[1], self.cells[0]))
        parts = [vel]
        if self.config.vorticity_in_state:
            # Do not compute manually, ask Cubism to compute it instead.
            self.sim.compute_vorticity()
            tmp = self.sim.fields.tmp.to_uniform(fill=0.0)  # (y, x)
            parts.append(tmp[np.newaxis])  # (1, y, x)
        if self.config.pressure_in_state:
            p = self.sim.fields.pres.to_uniform(fill=0.0)  # (y, x)
            parts.append(p[np.newaxis])    # (1, y, x)

        parts = [part.astype(self.dtype_np) for part in parts]
        out = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return out[np.newaxis]  # (1, 2 or 3 or 4, y, x)

    def fill_mock_state(self, state: np.ndarray):
        state[:, :, :, :] = np.where(np.arange(self.cells[0]) % 8 < 4, 0.5, -0.5)
        state[:, :, :, :] *= 1 - self.sim.fields.chi.to_uniform(fill=0.0)

    def get_mock_state(self):
        num_channels = 3 if self.config.vorticity_in_state else 2
        if self.config.pressure_in_state:
            num_channels += 1
        state = np.zeros((1, num_channels, *self.cells[::-1]), dtype=self.dtype_np)
        self.fill_mock_state(state[:, :2, :, :])
        if self.config.vorticity_in_state:
            h = self.config.extent / max(self.config.cells)
            compute_vorticity_2d(state[0, :2, :, :], self.h, fill=0.0,
                                 out=state[0, 2, :, :])
        if self.config.pressure_in_state:
            self.fill_mock_state(state[:, -1:, :, :])
        return TensorCollection(layers=self._downscale(state))

    def update_state_impl(self, import_state: np.ndarray, skip_steps):
        print(f"SKIP {skip_steps} STEPS\n", end="", file=sys.stderr, flush=True)
        self.step += skip_steps

        assert import_state.shape == self.import_state_shape, \
                (import_state.shape, self.import_state_shape)

        new_v = import_state[:, :2, :, :]
        assert new_v.shape == (1, 2, self.cells[1], self.cells[0]), new_v.shape
        new_v = np.moveaxis(new_v[0], 0, -1)  # (cells[1], cells[0], 2)
        if self.enforce_v_mask is not None:
            # Hard-code velocities under the obstacle to obstacle's velocity.
            shapes = self.sim.data.shapes
            if len(shapes) == 0:
                raise RuntimeError("enforce_v_mask incompatible with 0 shapes")
            elif len(shapes) > 1:
                raise NotImplementedError(
                        "enforce_v_mask with more than one shape not supported")
            new_v[self.enforce_v_mask] = (shapes[0].u, shapes[0].v)
        new_v = new_v.astype(self.dtype_cup)
        new_v = np.ascontiguousarray(new_v)
        self._data.time += skip_steps * self.config.dt_macro

        # dump = self.dump_state
        vel = self._data.vel

        # Load and adapt multiple times, because the current mesh likely does
        # not match the current state.
        # dump(prefix=f'onupdate{self.step:05d}0')
        vel.load_uniform(new_v)
        # dump(prefix=f'onupdate{self.step:05d}1')
        self.sim.adapt_mesh()
        print("Adapt #1 num blocks:", len(self.sim.fields.chi.blocks), file=sys.stderr)
        # dump(prefix=f'onupdate{self.step:05d}2')
        vel.load_uniform(new_v)
        # dump(prefix=f'onupdate{self.step:05d}3')
        self.sim.adapt_mesh()
        print("Adapt #2 num blocks:", len(self.sim.fields.chi.blocks), file=sys.stderr)
        # dump(prefix=f'onupdate{self.step:05d}4')
        vel.load_uniform(new_v)
        # dump(prefix=f'onupdate{self.step:05d}5')
        # for i in range(10):
        #     self.push_dump('onupdate')

        print("UPDATE STATE", flush=True)
        if self.config.pressure_in_state:
            # Even though mathematically the state is fully determined by the
            # velocity, we have to update the pressure field as well.
            # Otherwise, a get_state() invoked immediately after update_state()
            # and before advance() will return an outdated pressure field.
            # Furthermore, the old pressure is used as the initial guess in the
            # Poisson solver.
            assert import_state.shape[:2] == (1, 3)
            new_p = import_state[0, -1, :, :].astype(self.dtype_cup)
            self._data.pres.load_uniform(new_p)
            self._data.pold.load_uniform(new_p)  # Just in case.
