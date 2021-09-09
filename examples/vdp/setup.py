"""AdeLED with 5 LSTMs and no autoencoder for the Van der Pol oscillator."""

from adaled.utils.dataclasses_ import DataclassMixin, dataclass, field
import adaled
import adaled.utils.data.collections as data_collections

import numpy as np
import torch

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import os
import shlex
import sys

_FloatLike = Union[float, Sequence[float]]

DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_TEMPLATE = f'''\
#!/bin/bash

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {DIR}/{{SCRIPT}}.py "$@"
'''

@dataclass
class Config(DataclassMixin):
    # System.
    dt_micro: float = 0.001
    dt_rnn: float = 0.10
    num_parallel_sim: int = 1
    mu: adaled.ForcingConfig = field(lambda: adaled.ForcingConfig(
            kind='levels-cyclic', values=[3.0, 1.0],
            T=5000.0, decay_T=1e-50, brownian=0.0))
    circular_motion_system: bool = False

    # Dataset.
    train_capacity: int = 1280  # Such that 0.1*1280 % batch_size == 0.
    valid_capacity: int = 100
    # Length of trajectories in the dataset. The dataset currently stores
    # fixed-size trajectories.
    trajectory_length: int = 33
    dataset_histograms: Dict[str, adaled.HistogramStatsConfig] = field(lambda: {})

    # Networks.
    # Input size is 2 or 3, depending on append_F.
    network: adaled.RNNConfig = field(lambda: adaled.RNNConfig(
            output_size=2, rnn_hidden_size=32, rnn_num_layers=3,
            has_sigma2=True, append_F=False))
    ensemble_size: int = 5

    # Training.
    ensemble_training: adaled.TrainingConfig = \
            field(lambda: adaled.TrainingConfig(lr=0.002, scheduler='none'))
    nll_weight: float = 1.0
    mse_weight: float = 1.0
    adversarial_eps: float = 0.0
    adversarial_mse_weight: Optional[float] = None
    led_trainer: adaled.SequentialLEDTrainerConfig = \
            field(lambda: adaled.SequentialLEDTrainerConfig(
                trajectories_count_policy=adaled.SampleCountPolicyConfig(
                    per_timestep=2.0),
            ))

    # AdaLED.
    criteria: adaled.SimpleCriteriaConfig = \
            field(lambda: adaled.SimpleCriteriaConfig(
                k_warmup=7,
                k_cmp=25,
                max_cmp_error=0.01,    # euclidean distance of 0.14
                max_uncertainty=0.01,  # "euclidean stddev" of 0.14
                max_macro_steps=(80, 120),
                max_micro_steps=(80, 120),
                num_relaxation_steps=0))
    led: adaled.AdaLEDConfig = field(lambda: adaled.AdaLEDConfig(
            validation_every=20,
            dump_dataset=adaled.DumpConfig(every=50, keep_last=0),
            dump_transformer=adaled.DumpConfig(every=50, keep_last=0),
            dump_macro=adaled.DumpConfig(every=50, keep_last=0),
            log_every=1000,
            max_steps=401010,
            init_output_folder=False,
            # Micro is cheap enough, run always to simplify postprocessing.
            always_run_micro=True))
    recorder: adaled.RecorderConfig = field(lambda: adaled.RecorderConfig(
            start_every=2000, num_steps=2000, every=1, posttransform='float32',
            path_fmt='record-{start_timestep:07d}.pt'))

    def fix(self):
        """Fix arguments depending on other arguments."""
        if not self.network.has_sigma2:
            dh = self.dataset_histograms
            if 'latest_loss_macro_mse' in dh and 'latest_loss_macro_nll' in dh:
                dh['latest_loss_macro_mse'].data = \
                        ('metadata', 'latest_loss', 'macro')
                del dh['latest_loss_macro_nll']
            _kwargs = self.led_trainer.trajectories_sampling_policy.kwargs
            _kwargs['key'] = tuple(k for k in _kwargs['key'] if k != 'mse')

        self.network.input_size = 3 if self.network.append_F else 2


def ode_vdp_oscillator(t, X, mu: _FloatLike):
    """Van der Pol oscillator:

    dx/dt = mu * (x - 1/3 x^3 - y)
    dy/dt = 1/mu * x

    We use notation from Wikipedia:
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator

    Similar notation with eps instead mu is used in the original paper and in
    the Scholarpedia article.

    Balth Van der Pol, 'On "relaxation-oscillations"' (1926)
    https://www.tandfonline.com/doi/pdf/10.1080/14786442608564127

    http://scholarpedia.org/article/Van_der_Pol_oscillator

    Alternative previously used notation (mu -> 1/eps, y -> -y):
    https://authors.library.caltech.edu/27186/1/Caltech_ACM_TR_2009_04.pdf
    """
    x = X[..., 0]
    y = X[..., 1]
    return np.stack([
        mu * (x - (1. / 3) * x * x * x - y),
        x / mu,
    ], axis=-1)


def ode_circular_motion(x, T, r: _FloatLike, alpha: _FloatLike = 1.0):
    """System with damping towards a circular motion of given radius.

    dx/dt = -y - alpha * (x^2 + y^2 - r^2)
    dy/dt =  x - alpha * (x^2 + y^2 - r^2)
    """
    x = X[..., 0]
    y = X[..., 1]
    shift = _alpha * (x * x + y * y - self._r2)
    return np.stack([
        -y - x * shift,
        x - y * shift,
    ], axis=-1)


class MicroSolver(adaled.ForwardEulerMicroSolver):
    def __init__(
            self,
            ics: np.ndarray,
            circular: bool,
            dt_macro: float,
            dt_micro: float):
        if circular:
            def rhs(t, x, F):
                return ode_circular_motion(t, x, r=F)

        else:
            def rhs(t, x, F):
                return ode_vdp_oscillator(t, x, mu=F)

        super().__init__(rhs=rhs, dt=dt_macro, integrator_dt=dt_micro, x=ics)


def make_solver_and_trainer(config: Config):
    rnns = [adaled.RNN(config.network) for _ in range(config.ensemble_size)]
    adaled.print_model_info(rnns[0])

    def make_loss(mse_weight: Optional[float] = 1.0):
        if config.network.has_sigma2:
            return adaled.ProbabilisticLosses(
                    config.nll_weight, mse_weight,
                    detach_mu=config.network.detach_sigma2)
        else:
            loss = adaled.MSELosses()
            if mse_weight != 1.0:
                return adaled.nn.loss.ScaledLosses(loss, mse_weight)
            else:
                return loss

    kwargs = {}
    if config.network.has_sigma2:
        macro = adaled.ProbabilisticPropagatorEnsemble(rnns)
    else:
        macro = adaled.DeterministicPropagatorEnsemble(rnns)
    kwargs['loss'] = make_loss(config.mse_weight)

    if config.adversarial_eps > 0 \
            and (config.network.has_sigma2 or (config.adversarial_mse_weight or 0) > 0):
        kwargs['trainer_cls'] = adaled.RNNTrainerWithAdversarialLoss
        kwargs['adversarial_eps'] = config.adversarial_eps
        kwargs['adversarial_loss'] = make_loss(config.adversarial_mse_weight)
    else:
        kwargs['trainer_cls'] = adaled.RNNTrainer

    print("Loss:", kwargs['loss'])
    print("Adversarial loss:", kwargs.get('adversarial_loss'))
    trainer = macro.make_trainer(config.ensemble_training, **kwargs)
    return (macro, trainer)


def main(config: Config):
    print(' '.join(map(shlex.quote, sys.argv)))
    config.fix()
    config.pretty_print()
    config.validate()
    adaled.set_highlighted_excepthook()
    adaled.init_torch()  # Enable double precision for consistency with numpy.
    adaled.save(config, 'config.pt')
    adaled.save_executable_script('plot.sh', SCRIPT_TEMPLATE.format(SCRIPT='plotting'))
    adaled.save_executable_script('postprocess.sh', SCRIPT_TEMPLATE.format(SCRIPT='postprocess'))
    adaled.save_executable_script('movie.sh', SCRIPT_TEMPLATE.format(SCRIPT='movie'))

    mu_func = config.mu.make_func(config.led.max_steps, config.dt_rnn,
                                  num_simulations=config.num_parallel_sim)
    # mu_func.debug()
    # return

    macro, ensemble_trainer = make_solver_and_trainer(config)
    transformer = adaled.IdentityTransformer()

    trajectory_datasets = data_collections.CappedFixedLengthTrajectoryDatasetCollection(
            config.train_capacity,
            config.valid_capacity,
            config.trajectory_length)
    trainer = config.led_trainer.make(
            macro_trainer=ensemble_trainer, transformer=transformer)

    # LED specification.
    led = adaled.AdaLED(
            macro=macro,
            trainer=trainer,
            transformer=transformer,
            config=config.led,
            criteria=config.criteria.create(),
            datasets=trajectory_datasets)
    led.server.diagnostics.add_dataset_histograms(config.dataset_histograms)

    # System and solvers specification.
    ics = np.random.uniform(-5.0, 5.0, (config.num_parallel_sim, 2))
    micro = MicroSolver(ics, circular=config.circular_motion_system,
                        dt_macro=config.dt_rnn, dt_micro=config.dt_micro)

    # Pass external forcing to AdaLED for postprocessing purposes, the network
    # will ignore it depending on the `append_F` config.
    external_forcing_func = adaled.utils.misc.function_to_generator(
            lambda t: np.full(config.num_parallel_sim, mu_func(t)),
            config.dt_rnn)
    generator = led.make_generator(micro, external_forcing_func)
    led.run(generator, config.recorder)
