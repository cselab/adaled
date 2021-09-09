from ..common.config import AutoencoderConfig, CombinedConfigBase, \
        DatasetConfig, ScalingConfig
from ..common.setup import SetupBase2D
from .micro import CylinderMicroConfig, CUP2DFlowBehindCylinderSolver

import adaled
from adaled.transformers.multiresolution import Multiresolution, sigmoid
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass, field

from examples.cubismup.common.config import CUPAdaLEDConfig

import numpy as np
import torch

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import os

def _default_histograms() -> Dict[str, adaled.HistogramStatsConfig]:
    return {
        'F': adaled.HistogramStatsConfig(15, (50.0, 1550.0), ('trajectory', 'F')),
        # 'mse' part removed when has_sigma2 == False.
        'latest_loss_macro_mse': adaled.HistogramStatsConfig(
                25, (1e-6, 1e2), ('metadata', 'latest_loss', 'macro', 'mse'), log=True),
        # Hierarchy added later.
        'latest_loss_transformer': adaled.HistogramStatsConfig(
                25, (1e-6, 1e2), ('metadata', 'latest_loss', 'transformer'), log=True),
    }


@dataclass
class Config(CombinedConfigBase):
    """
    Attributes:
        cyl_weight: (float) extra weight sigmoid-like factor to assign to the
                region near the cylinder boundary when computing reconstruction
                loss, defaults to 1.0 (no extra weight)
        cyl_weight_distance: (float) parameter of the weight factor sigmoid
        cyl_weight_sigma: (float) parameter of the weight factor sigmoid
    """
    # Change every ~20k steps, add some smoothing to avoid spikes in pressure
    # and viscous forces.
    Re: adaled.ForcingConfig = field(lambda: adaled.ForcingConfig(
            kind='levels-cyclic', values=[500.0, 1000.0],
            T=100.0, decay_T=1e-50, smooth_T=0.04, brownian=0.0))

    # Maximum Reynolds number, for scaling before plugging into the RNNs. It
    # should not automatically computed, as it could lead in future to
    # erroneously having different dt and scaling on different ranks.
    max_Re: float = 1000.0

    cyl_weight: float = 1.0
    cyl_weight_distance: float = 0.005
    cyl_weight_sigma: float = 0.001

    # Forces seem to reach up to 0.13 for Re=1000 and 0.03 for Re=1000.
    # Scaling of 10x gets the values closer to 1 (F_t - F_{t-1} is about 0.004).
    # Convolutional autoencoder produces values of +- 0.8.
    cyl_forces_ae_scaling: float = 10.0

    # override
    autoencoder: AutoencoderConfig = field(lambda: AutoencoderConfig(
       scaling=ScalingConfig(potential_margin=100.0),
    ))

    # override
    dataset: DatasetConfig = field(lambda: DatasetConfig(
        train_capacity=256,  # Global capacity.
        valid_capacity=24,   # Global capacity.
    ))

    # override
    micro: CylinderMicroConfig = field(lambda: CylinderMicroConfig(
        cells=(256, 128),
        dt_macro=0.005,  # ~120 steps per period for Re=500, ~60 for Re=1000
    ))

    # override
    led: CUPAdaLEDConfig = field(lambda: CUPAdaLEDConfig(
        dataset_histograms=_default_histograms(),
    ))

    def compute_max_v(self):
        """Compute maximum cylinder velocity from the maximum Reynolds number."""
        return self.micro.Re_to_vel(self.max_Re)

    def extra_latent_size(self) -> int:
        """Compute the size of the quantities of interest vector (forces on cylinder)."""
        if self.micro.forces_in_state == 'separate':
            return 4  # Fx_P, Fy_P, Fx_V, Fy_V
        elif self.micro.forces_in_state == 'total':
            return 2  # Fx_P + Fx_V, Fy_P + Fy_V
        else:
            return 0

    def make_F_func(self):
        return self.Re.make_func(self.led.client.max_steps, dt=self.micro.dt_macro)

    def make_scaling_F(self):
        return adaled.Scaling(min=0.0, max=1.2 * self.max_Re)

    # override
    def compute_multiresolution_submask_weights(
            self, mr: Multiresolution) -> List[Optional[np.ndarray]]:
        """Modify default layer weights with extra prioritized weight around
        the cylinder."""
        weights = super().compute_multiresolution_submask_weights(mr)
        if self.cyl_weight == 1 or dim_mask is None:
            return weights

        out = []
        h = self.micro.compute_h()
        for i, (weight, layer) in enumerate(zip(weights, mr.layers)):
            centers = h * layer.compute_downscaled_cell_centers()
            dist = np.sqrt(((centers - np.array(self.micro.center)) ** 2).sum(axis=-1))
            sdf = dist - self.micro.r

            weight_scale = 1 + (self.cyl_weight - 1) * sigmoid(
                    (self.cyl_weight_distance - np.abs(sdf)) / self.cyl_weight_sigma)
            if weight is not None:
                assert weight.shape == sdf.shape, (weight.shape, sdf.shape)
                weight *= weight_scale
            else:
                weight = weight_scale
            out.append(weight)

        return out


class Setup(SetupBase2D):
    MOVIE_SCRIPT_PATH = '-m examples.cubismup.common.movie_2d'
    PLOTTING_SCRIPT_PATH = '-m examples.cubismup.cyl2d.plotting'
    POSTPROCESS_SCRIPT_PATH = '-m examples.cubismup.cyl2d.postprocess'

    def make_micro(self, F_func, comm):
        config = self.config
        return CUP2DFlowBehindCylinderSolver(
                config.micro, Re_func=F_func, max_Re=config.max_Re, comm=comm)

    def make_transformer_hierarchy(self, *args, **kwargs):
        hierarchy = super().make_transformer_hierarchy(*args, **kwargs)

        if self.config.micro.forces_in_state:
            ae = adaled.Scaling(scale=self.config.cyl_forces_ae_scaling, shift=0.0)
            hierarchy['cyl_forces'] = \
                    (ae.to_transformer(), self.config.extra_latent_size())
        return hierarchy
