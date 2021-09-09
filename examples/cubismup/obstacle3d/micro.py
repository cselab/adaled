from ..common.config import Micro3DConfigBase, SymmetryBreakingConfig
from ..common.micro_3d import CUP3DSolver, SetObstacleVelocityOperator3D, cup3d
from adaled.transformers.multiresolution import MRLayerConfig
from adaled.utils.dataclasses_ import SPECIFIED_LATER, dataclass, field

from typing import Callable, List, Tuple

@dataclass
class ObstacleMicroConfig(Micro3DConfigBase):
    # override
    multiresolution: List[MRLayerConfig] = field(lambda: [
        MRLayerConfig(stride=2),
        MRLayerConfig(center=(0.21, 0.25, 0.25), size=(0.25, 0.25, 0.25),
                      alpha_margin_cells=12.0, alpha_sigma_cells=2.0),
    ])

    kind: str = 'Sphere'
    # L: float = 0.0375 * 2
    L: float = 0.05 * 2
    center: Tuple[float, float, float] = (0.20, 0.25, 0.25)
    symmetry_breaking: SymmetryBreakingConfig = \
            field(lambda: SymmetryBreakingConfig(dv=(0.0, 0.1, 0.0)))

    def Re_to_vel(self, Re: float) -> float:
        return Re * self.nu / self.L

    def get_factory_content(self):
        kind, L, center = self.kind, self.L, self.center
        return f'{kind} L={L} bForcedInSimFrame=1 bFixFrameOfRef=1 ' \
               f'xpos={center[0]} ypos={center[1]} zpos={center[2]} '


class CUP3DFlowBehindObstacleSolver(CUP3DSolver):
    def __init__(self,
                 config: ObstacleMicroConfig,
                 Re_func: Callable[[float], float],
                 max_Re: float,
                 **kwargs):
        max_v = config.Re_to_vel(Re=max_Re)
        self.Re_func = Re_func
        super().__init__(config=config, max_v=max_v,
                         factory_content=config.get_factory_content(), **kwargs)

    def add_obstacles_and_operators(self, sim: cup3d.Simulation):
        obstacles = sim.obstacles
        if len(obstacles) == 0:
            raise TypeError("expected at least one obstacle")

        def uvw_func(t: float):
            Re = self.Re_func(t)
            return (-self.config.Re_to_vel(Re), 0.0, 0.0)

        self._op = SetObstacleVelocityOperator3D(
                sim, obstacles[0], uvw_func, self.config.symmetry_breaking)
        sim.insert_operator(self._op)
