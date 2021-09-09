from ..common.config import Micro2DConfigBase
from ..common.micro_2d import CUP2DSolver, SymmetryBreakingOperator, cup2d
from adaled.backends import TensorCollection, get_backend
from adaled.transformers.multiresolution import MRLayerConfig, Multiresolution
from adaled.utils.dataclasses_ import SPECIFIED_LATER, dataclass, field

import numpy as np

from typing import Callable, List, Sequence, Tuple
import math

_DEFAULT_LAYER_KWARGS = {
    # For __post_init__...
    None: {'alpha_margin_cells': SPECIFIED_LATER, 'alpha_sigma_cells': SPECIFIED_LATER},
    # Depending on the resolution:
    (256, 128): {'alpha_margin_cells': 12.0, 'alpha_sigma_cells': 1.5},
    (1024, 512): {'alpha_margin_cells': 22.0, 'alpha_sigma_cells': 3.0},
    (2048, 1024): {'alpha_margin_cells': 22.0, 'alpha_sigma_cells': 3.0},
}

@dataclass
class CylinderMicroConfig(Micro2DConfigBase):
    # override
    multiresolution: List[MRLayerConfig] = SPECIFIED_LATER

    r: float = 0.0375
    center: Tuple[float, float] = (0.20, 0.25)

    # If non-empty, macro propagator will be configured to predict the
    # forces. The forces are anyway always recorded, even if this is False.
    # Possible values: '', 'total', 'separate'
    forces_in_state: str = ''

    # Parameters of the symmetry breaking.
    sym_breaking_omega: float = 1 / (1 * 0.005)
    sym_breaking_decay: float = 1 / (2 * 0.005)

    def __post_init__(self):
        if self.multiresolution == SPECIFIED_LATER:
            # Note: self.cells is not known yet (argv not parsed yet), so only
            # fill with SPECIFIED_LATER.
            kwargs = _DEFAULT_LAYER_KWARGS[None]
            self.multiresolution = [
                MRLayerConfig(stride=2, **kwargs),
                MRLayerConfig(center=(0.20, 0.25), size=(0.25, 0.25), **kwargs),
            ]

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'sym_breaking_omega' not in state:
            self.sym_breaking_omega = 1 / (1 * 0.005)
            self.sym_breaking_decay = 1 / (2 * 0.005)

    def fix(self, round_sizes_to: List[int]):
        defaults = _DEFAULT_LAYER_KWARGS.get(tuple(self.cells), {})
        for layer in self.multiresolution:
            for key, value in defaults.items():
                if getattr(layer, key) == SPECIFIED_LATER:
                    setattr(layer, key, value)

        super().fix(round_sizes_to)

    def validate(self, *args, **kwargs):
        super().validate(*args, **kwargs)
        assert self.forces_in_state in ['', 'separate', 'total'], self.forces_in_state

    def Re_to_vel(self, Re: float) -> float:
        return Re * self.nu / (2 * self.r)

    def record_qoi_to_state_qoi(
            self, record_qoi: TensorCollection) -> TensorCollection:
        """If macro is trained to predict total forces, compute total from
        separate forces (viscous + pressure)."""
        if self.forces_in_state == 'total':
            Fcyl = record_qoi['cyl_forces']
            Fpx = Fcyl[:, :, 0]
            Fpy = Fcyl[:, :, 1]
            Fvx = Fcyl[:, :, 2]
            Fvy = Fcyl[:, :, 3]
            forces = np.stack([Fpx + Fvx, Fpy + Fvy], axis=-1)
            out = TensorCollection(cyl_forces=forces)
        else:
            out = record_qoi
        return out

    # override
    def compute_obstacle_interior_mask(self) -> np.ndarray:
        x, y = self.compute_cell_centers()
        dx = x - self.center[0]
        dy = y - self.center[1]
        dx2 = np.broadcast_to((dx * dx)[np.newaxis, :], self.cells[::-1])
        dy2 = np.broadcast_to((dy * dy)[:, np.newaxis], self.cells[::-1])
        d2 = dx2 + dy2
        return d2 < (self.r + self.enforce_obstacle_velocity_sdf) ** 2


class CUP2DFlowBehindCylinderSolver(CUP2DSolver):
    def __init__(self,
                 config: CylinderMicroConfig,
                 Re_func: Callable[[float], float],
                 max_Re: float,
                 **kwargs):
        max_v = config.Re_to_vel(Re=max_Re)
        self.Re_func = Re_func
        super().__init__(config=config, max_v=max_v, **kwargs)

    def add_obstacles_and_operators(self, sim: cup2d.Simulation):
        config: CUP2DFlowBehindCylinderConfig = self.config
        r = config.r
        self._disk = cup2d.Disk(
                sim, r=r, center=config.center,
                vel=(config.Re_to_vel(Re=self.Re_func(0.0)), 0.0),
                fixedx=True, fixedy=True,  # fixedy may be True.
                forcedx=True, forcedy=True)
        sim.add_shape(self._disk)

        def uv_func(t: float):
            Re = self.Re_func(t)
            return (-self.config.Re_to_vel(Re), 0.0)

        sim.insert_operator(SymmetryBreakingOperator(
                sim, self._disk, uv_func, dy=0.1 * r,
                omega=self.config.sym_breaking_omega,
                decay=self.config.sym_breaking_decay))

    def get_quantities_of_interest(self, for_state: bool) -> dict:
        """Return a dictionary `{'cyl_forces': 1x4 array}` or an empty dictionary."""
        assert self.config.forces_in_state in ['', 'separate', 'total']
        if for_state and not self.config.forces_in_state:
            return {}

        Fp = self._disk.force_P
        Fv = self._disk.force_V
        if for_state and self.config.forces_in_state == 'total':
            forces = [Fp[0] + Fv[0], Fp[1] + Fv[1]]
        else:
            forces = [Fp[0], Fp[1], Fv[0], Fv[1]]
        forces = np.array([forces], dtype=self.dtype_np)  # (1, 2 or 4)
        return {'cyl_forces': forces}
