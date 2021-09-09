from .config import Micro3DConfigBase, SymmetryBreakingConfig
from .micro import CUPSolverBase
from .utils_3d import curl
from adaled import TensorCollection
from adaled.utils.dataclasses_ import DataclassMixin, dataclass

import libcubismup3d as cup3d
import numpy as np
import torch

from typing import Callable, List, Optional, Tuple, TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from mpi4py import MPI

class SetObstacleVelocityOperator3D(cup3d.Operator):
    def __init__(
            self,
            sim: cup3d.Simulation,
            obstacle: cup3d.Obstacle,
            uvw_func: Callable[[float], Tuple[float, float, float]],
            symmetry_breaking: Optional[SymmetryBreakingConfig] = None):
        super().__init__(sim.data, self.__class__.__name__)
        self.obstacle = obstacle
        self.uvw_func = uvw_func
        self._sym = symmetry_breaking

    def __call__(self, dt: float):
        t = self.data.time
        uvw = self.uvw_func(t)
        if self._sym:
            uvw = self._sym.modify_velocity(uvw, t)
        uvw = tuple(float(v) for v in uvw)
        if self.data.step % 10 == 0:
            print(f"Step #{self.data.step}: setting velocity of "
                  f"{self.obstacle} to {uvw}", flush=True)
        self.obstacle.v_imposed = uvw


class CUP3DSolver(CUPSolverBase):
    dtype_cup = np.float64

    def init_simulation(self,
                        config: Micro3DConfigBase,
                        dt_micro: float,
                        comm: 'MPI.Intracomm',
                        *,
                        factory_content: str,
                        extra_argv: List[str] = []) -> cup3d.Simulation:
        nlevels, cup_cells = config.compute_nlevels_and_cup_cells()
        argv = [
            '-bpdx', cup_cells[0] // cup3d.BLOCK_SIZE[0],
            '-bpdy', cup_cells[1] // cup3d.BLOCK_SIZE[1],
            '-bpdz', cup_cells[2] // cup3d.BLOCK_SIZE[2],
            '-extentx', config.extent,
            '-extenty', -1.0,
            '-extentz', -1.0,
            '-nsteps', 0,
            '-nu', config.nu,
            '-cfl', 0.0,  # dt manually computed.
            '-dt', dt_micro,
            '-factory-content', factory_content,
            '-levelStart', 0,
            '-levelMax', nlevels,
            '-verbose', int(config.verbose),
            '-Ctol', 0.1,
            '-Rtol', 0.5,
            '-rampup', config.rampup_steps,
            *extra_argv,
            *config.extra_cubism_argv,
        ]
        if config.output_dir != '.':
            raise NotImplementedError(config.output_dir)
        argv = [str(arg) for arg in argv]
        if comm is not None:
            from mpi4py import MPI
            comm = MPI._addressof(comm)
        else:
            comm = 0
        sim = cup3d.Simulation(argv, comm)
        self._data: cup3d.SimulationData = sim.data
        self.add_obstacles_and_operators(sim)
        return sim

    # virtual
    def add_obstacles_and_operators(self, sim: cup3d.Simulation):
        pass

    def advance_impl(self, cup_steps: int):
        self._data.nsteps += cup_steps
        self.sim.run()

    def dump_state(self, prefix: str):
        # self._data.dump_tmp(prefix)
        # self._data.dump_vel(prefix)
        raise NotImplementedError("dumping 3D state not implemented")

    def get_local_state(self) -> np.ndarray:
        """Return the state as an ndarray of shape (2 or 3, [grid_z], grid_y, grid_x),
        where the first dimension represents the channels vx, vy and optionally
        vorticity."""
        if self.config.pressure_in_state:
            raise NotImplementedError("pressure_in_state not implemented for 3D")
        # Fill with zeros, in order to sum up results from different ranks.
        num_channels = self.config.get_num_export_channels()
        out = np.empty((num_channels,) + self.cells[::-1])  # (3 or 4 or 6 or 7, z, y, x)
        fields = self.sim.fields
        fields.u.to_uniform(out=out[0])
        fields.v.to_uniform(out=out[1])
        fields.w.to_uniform(out=out[2])
        if self.config.vorticity_in_state:
            self.sim.compute_vorticity()
            fields.tmpU.to_uniform(out=out[3])
            fields.tmpV.to_uniform(out=out[4])
            fields.tmpW.to_uniform(out=out[5])
        if self.config.pressure_in_state:
            fields.p.to_uniform(out=out[-1])

        if self.dtype_np == np.float32:
            # Tensor's float64 -> float32 conversion is several times faster
            # than that of numpy. Possibly due to multithreading.
            # out = out.astype(self.dtype_np)
            out = torch.from_numpy(out).float().numpy()

        return out[np.newaxis]  # (1, 3 or 4 or 6 or 7, z, y, x)

    def fill_mock_state(self, state: np.ndarray):
        half = self.dtype_np(0.5)
        state[:, :, :, :, :] = np.where(np.arange(self.cells[0]) % 8 < 4, +half, -half)
        chi = np.empty(state.shape[2:], dtype=self.dtype_cup)
        self.sim.fields.chi.to_uniform(out=chi)
        chi = chi.astype(self.dtype_np)
        np.subtract(1, chi, out=chi)  # chi = 1 - chi
        state[:, :, :, :, :] *= chi  # *= 1 - chi

    def get_mock_state(self):
        num_channels = self.config.get_num_export_channels()
        state = np.zeros((1, num_channels, *self.cells[::-1]), dtype=self.dtype_np)
        self.fill_mock_state(state[:, :3, :, :, :])
        if self.config.vorticity_in_state:
            curl(state[:, :3, :, :, :], self.config.compute_h(),
                 out=state[:, 3:6, 1:-1, 1:-1, 1:-1])
        if self.config.pressure_in_state:
            self.fill_mock_state(state[:, -1:, :, :, :])
        return TensorCollection(layers=self._downscale(state))

    def update_state_impl(self, merged: np.ndarray, skip_steps):
        print(f"SKIP {skip_steps} STEPS\n", end="", file=sys.stderr, flush=True)
        self.step += skip_steps

        assert merged.shape == self.import_state_shape, \
                (merged.shape, self.import_state_shape)

        new_v = merged[:, :3, :, :, :]
        assert new_v.shape == (1, 3, *self.cells[::-1]), (new_v.shape, self.cells)
        assert new_v.data.c_contiguous
        self._data.time += skip_steps * self.config.dt_macro

        fields = self.sim.fields
        def load():
            fields.u.load_uniform(new_v[0, 0, :, :, :].astype(self.dtype_cup))
            fields.v.load_uniform(new_v[0, 1, :, :, :].astype(self.dtype_cup))
            fields.w.load_uniform(new_v[0, 2, :, :, :].astype(self.dtype_cup))

        # Load and adapt multiple times, because the current mesh likely does
        # not match the current state.
        load()
        self.sim.adapt_mesh()
        load()
        self.sim.adapt_mesh()
        load()

        if self.config.pressure_in_state:
            new_p = merged[0, -1, :, :, :]
            fields.p.load_uniform(new_p.astype(self.dtype_cup))
