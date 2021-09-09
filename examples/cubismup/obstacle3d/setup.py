from ..common.config import AutoencoderConfig, CombinedConfigBase, \
        DatasetConfig, LayerLossConfig, ScalingConfig
from ..common.setup import SetupBase3D
from .micro import ObstacleMicroConfig, CUP3DFlowBehindObstacleSolver

import adaled
from adaled.utils.dataclasses_ import dataclass, field

from examples.cubismup.common.config import CUPAdaLEDConfig, DatasetConfig

import numpy as np

from typing import Dict
import copy

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
    Re: adaled.ForcingConfig = field(lambda: adaled.ForcingConfig(
            kind='levels-cyclic', values=[300.0, 350.0, 400.0],
            T=200.0, decay_T=1e-50, brownian=0.0))  # Change every ~10k steps.

    # Do not automatically compute max Re, because a different value might be
    # used by different ranks if brownian != 0.0 and ranks seed random
    # ForcingConfig differently.
    max_Re: float = 400.0

    # override
    autoencoder: AutoencoderConfig = field(lambda: AutoencoderConfig(
        loss=LayerLossConfig(layer_margin_cells=6),
        scaling=ScalingConfig(potential_margin=500.0),
    ))

    # override
    dataset: DatasetConfig = field(lambda: DatasetConfig(
        train_capacity=256,  # Global capacity.
        valid_capacity=24,   # Global capacity.
    ))

    # override
    micro: ObstacleMicroConfig = field(lambda: ObstacleMicroConfig(
        cells=(256, 128, 128),
        dt_macro=0.020,
    ))

    # override
    led: CUPAdaLEDConfig = field(lambda: CUPAdaLEDConfig(
        dataset_histograms=_default_histograms(),
        criteria=adaled.SimpleCriteriaConfig(
                k_warmup=4,
                k_cmp=12,
                max_cmp_error=0.00001,  # Mean square error in real space!
                max_uncertainty=0.01,   # Mean square error in latent space!
                max_macro_steps=(80, 120),
                # Keep trajectories sent from clients to server as short as
                # possible! Use only up to 1 step to add a bit of randomness.
                max_micro_steps=(0, 1),
                num_relaxation_steps=10),
        led_trainer=adaled.SequentialLEDTrainerConfig(
                macro_batch_size=8,
                transformer_batch_size=2,
                states_count_policy=adaled.SampleCountPolicyConfig(fixed=8),
                trajectories_count_policy=adaled.SampleCountPolicyConfig(fixed=8)),
    ))

    def compute_max_v(self):
        return self.micro.Re_to_vel(self.max_Re)

    def make_F_func(self):
        return self.Re.make_func(self.led.client.max_steps, dt=self.micro.dt_macro)

    def make_scaling_F(self):
        return adaled.Scaling(min=0.0, max=1.2 * self.max_Re)


class Setup(SetupBase3D):
    MOVIE_SCRIPT_PATH = '-m examples.cubismup.common.movie_3d'
    PLOTTING_SCRIPT_PATH = '-m examples.cubismup.obstacle3d.plotting'

    def make_micro(self, F_func, comm):
        config = self.config
        return CUP3DFlowBehindObstacleSolver(
                config.micro, Re_func=F_func, max_Re=config.max_Re, comm=comm)
