from adaled.nn.loss import weighted_mse_losses
from adaled.utils.dataclasses_ import dataclass, field
import adaled
import adaled.utils.data.collections as data_collections

import numpy as np
import torch

from typing import Dict, List, Tuple, Sequence
import shlex
import sys


SCRIPT_TEMPLATE = '''\
#!/bin/bash

export CUDA_VISIBLE_DEVICES=

OMP_NUM_THREADS=1 python -m examples.reaction_diffusion.{SCRIPT} "$@"
'''


@dataclass
class RDConfig(adaled.DataclassMixin):
    cells: Tuple[int, int] = (96, 96)  # (y, x)
    L: float = 20.0
    dt_macro: float = 0.05
    steps_per_macro: int = 1
    beta: float = 1.0


def diffusion_term(uv: np.ndarray, hx: float, hy: float, D: np.ndarray):
    assert uv.ndim == 3, uv.shape  # (channel, row, col)
    assert D.shape == (2,), D.shape
    assert D.shape[0] == uv.shape[0], (D.shape, uv.shape)

    out = np.zeros_like(uv)

    factor_x = 1 / (hx * hx)
    factor_y = 1 / (hy * hy)

    out[:, :, 1:]   = factor_x * (uv[:, :, :-1] - uv[:, :, 1:])
    out[:, :, :-1] += factor_x * (uv[:, :, 1:] - uv[:, :, :-1])
    out[:, 1:, :]  += factor_y * (uv[:, :-1, :] - uv[:, 1:, :])
    out[:, :-1, :] += factor_y * (uv[:, 1:, :] - uv[:, :-1, :])
    out *= D[:, None, None]
    return  out


def rk45_step(rhs, x, t, dt):
    # Keep the time parameter in case we need it later.
    k1 = rhs(x, t)
    k2 = rhs(x + dt/4*k1, t + dt/4)
    k3 = rhs(x + dt*(3/32)*k1 + dt*(9/32)*k2, t + 3/8*dt)
    k4 = rhs(x + dt*(1932/2197)*k1 - dt*(7200/2197)*k2 + dt*(7296/2197)*k3, t + 12/13*dt)
    k5 = rhs(x + dt*(439/216)*k1 - dt*8*k2 + dt*(3680/513)*k3 - dt*(845/4104)*k4, t + dt)
    k6 = rhs(x + dt*(-8/27)*k1 + dt*2*k2 - dt*(3544/2565)*k3 + dt*(1859/4104)*k4 - dt*(11/40)*k5, t + dt/2)

    out = x + dt*(25/216)*k1 + dt*(1408/2565)*k3 + dt*(2197/4104)*k4 - dt/5*k5
    return out


class ReactionDiffusionSolver(adaled.MicroSolver):
    def __init__(self, config: RDConfig):
        self.config = config

        self._hy = config.L / config.cells[0]
        self._hx = config.L / config.cells[1]

        self.uv = self.create_initial_conditions(config)
        self._step = 0
        assert self.uv.shape == (1, 2, *config.cells)

    @staticmethod
    def create_initial_conditions(config: RDConfig):
        NY, NX = config.cells
        L = config.L
        x = (-NX / 2 + 0.5 + np.arange(NX)) * (L / NX)
        y = (-NY / 2 + 0.5 + np.arange(NY)) * (L / NY)
        x, y = np.meshgrid(x, y)

        r = np.sqrt(x * x + y * y)
        phi = np.arctan2(y, x)

        uv = np.empty((1, 2, NY, NX))  # (batch, channels, y, x)
        uv[0, 0, :, :] = np.tanh(r * np.cos(phi - r))
        uv[0, 1, :, :] = np.tanh(r * np.sin(phi - r))

        return uv

    def advance(self, F: np.ndarray):
        assert F.shape == (1,)
        d1 = d2 = F[0]
        D = np.array([d1, d2])
        beta = self.config.beta
        self._time_step(D, beta)
        return self.get_state()

    def _time_step(self, D: np.ndarray, beta: float):
        def rhs(uv, t):
            u = uv[0, : ,:]
            v = uv[1, :, :]
            u2v2 = u * u + v * v
            one_u2v2 = 1 - u2v2

            duv_dt = np.zeros_like(uv)
            duv_dt[0, :, :] = one_u2v2 * u + beta * u2v2 * v
            duv_dt[1, :, :] = -beta * u2v2 * u + one_u2v2 * v
            duv_dt += diffusion_term(uv, self._hx, self._hy, D)
            return duv_dt

        dt = self.config.dt_macro / self.config.steps_per_macro
        # for i in range(self.config.steps_per_macro):
        #     self.uv[0, :, :, :] += dt * rhs(self.uv[0], 0.0)
        self.uv[0, :, :, :] = rk45_step(rhs, self.uv[0, :, :, :], 0.0, dt)

        self._step += 1
        if self._step % 100 == 0:
            print(f"STEP={self._step} mean={self.uv.mean()}  std={self.uv.std()}")

    def get_state(self):
        return self.uv.copy()

    def update_state(self, new_state: np.ndarray, skip_steps: int):
        uv = new_state
        if hasattr(uv, 'cpu'):
            uv = uv.cpu()
        assert uv.shape == self.uv.shape, (uv.shape, self.uv.shape)
        self.uv[:] = uv


@dataclass
class Config(adaled.DataclassMixin):
    rd: RDConfig = field(RDConfig)
    Fd: adaled.ForcingConfig = field(lambda: adaled.ForcingConfig(
            kind='levels-cyclic', values=[0.10, 0.20],
            T=100.0, decay_T=1e-50, smooth_T=0.0, brownian=0.0))

    # Dataset.
    train_capacity: int = 1024
    valid_capacity: int = 64
    trajectory_length: int = 24

    # Macro networks.
    # Input size is 2 or 3, depending on append_F.
    rnn: adaled.RNNConfig = field(lambda: adaled.RNNConfig(
            rnn_hidden_size=32, rnn_num_layers=2,
            has_sigma2=True, append_F=True))
    ensemble_size: int = 5

    # Macro training.
    ensemble_training: adaled.TrainingConfig = \
            field(lambda: adaled.TrainingConfig(lr=0.001, scheduler='none'))
    led_trainer: adaled.SequentialLEDTrainerConfig = field(lambda: adaled.SequentialLEDTrainerConfig(
            # Batch size per rank.
            macro_batch_size=8,
            transformer_batch_size=8,
            states_count_policy=adaled.SampleCountPolicyConfig(fraction=0.0625),
            trajectories_count_policy=adaled.SampleCountPolicyConfig(fraction=0.125)))

    # Autoencoder.
    scale: float = 1.1
    encoder: adaled.ConvMLPEncoderConfig = field(lambda: adaled.ConvMLPEncoderConfig(
            latent_state_dim=8,
            # Number of layers should be adjusted to the grid size.
            conv_layers_kernel_sizes=5,
            conv_layers_channels=[16, 16, 16, 16],
            conv_layers_strides=1,
            pool_kernel_sizes=2,
            pool_kernel_strides=2,
            padding_mode='replicate',  # or 'none'?
            activation='celu',
            activation_output='tanh',
            batch_norm=False))
    ae_training: adaled.TrainingConfig = field(lambda: adaled.TrainingConfig(lr=0.001))

    # AdaLED.
    criteria: adaled.SimpleCriteriaConfig = \
            field(lambda: adaled.SimpleCriteriaConfig(
                k_warmup=5,
                k_cmp=18,
                max_cmp_error=0.001,
                max_uncertainty=0.01,
                max_macro_steps=(400, 500),
                max_micro_steps=(10, 15),
                num_relaxation_steps=0))
    led: adaled.AdaLEDConfig = field(lambda: adaled.AdaLEDConfig(
            validation_every=20,
            dump_dataset=adaled.DumpConfig(every=30, keep_last=0),
            dump_transformer=adaled.DumpConfig(every=30, keep_last=0),
            dump_macro=adaled.DumpConfig(every=30, keep_last=0),
            log_every=100,
            max_steps=10003,
            init_output_folder=False,
            # Micro is cheap enough, run always to simplify postprocessing.
            always_run_micro=True))
    recorder: adaled.RecorderConfig = field(lambda: adaled.RecorderConfig(
            start_every=1000, num_steps=1000, every=1, x_every=10, posttransform='float32',
            path_fmt='record-{sim_id:03d}-{start_timestep:07d}.h5'))

    def fix(self):
        self.encoder.fix()

        F_size = 1  # d == d1 == d2
        self.encoder.input_shape = (2, *self.rd.cells)
        self.rnn.input_size = self.encoder.latent_state_dim + F_size
        self.rnn.output_size = self.encoder.latent_state_dim


class Recorder(adaled.TrajectoryRecorder):
    def __init__(self,
                 config: Config,
                 generator: adaled.AdaLEDGenerator,
                 extra_sim_records: Dict[str, adaled.DynamicArray] = {},
                 **kwargs):
        if config.led.always_run_micro:
            # See below for the meaning of validation elements.
            self.record_validation = adaled.DynamicArray()
            extra_sim_records = {
                **extra_sim_records,
                'validation': self.record_validation,
            }
        else:
            self.record_macro_validation = None
            self.record_macro_ae_validation = None
        super().__init__(config=config.recorder,
                         extra_sim_records=extra_sim_records, **kwargs)
        self.generator = generator
        self.cmp_error = generator.parent.criteria.error_func

    def record_step(self, i: int, relative_i: int, step: adaled.AdaLEDStep, *args, **kwargs):
        super().record_step(i, relative_i, step, *args, **kwargs)

        if self.record_validation is not None:
            with self.generator.measure_stats_overhead():
                if step.z is not None:
                    self.record_validation.append(self._compute_validation(step))
                else:
                    self.record_validation.append(np.nan)

    def _compute_validation(self, step: adaled.AdaLEDStep):
        """Compute various x and z validations."""
        assert step.z is not None
        transformer = self.generator.parent.transformer
        x_micro = torch.from_numpy(step.x)
        z_macro = step.z
        with torch.no_grad():
            z_micro = transformer.transform(x_micro).cpu()
            x_macro = transformer.inverse_transform(z_macro).cpu()
            error_z = adaled.to_numpy(adaled.nn.loss.mse_losses(z_macro.cpu(), z_micro))
            error_abs = weighted_mse_losses(
                    computed=x_macro, expected=x_micro, weight=None, relative=False)
            error_rel = weighted_mse_losses(
                    computed=x_macro, expected=x_micro, weight=None, relative=True)

            z_macro2 = transformer.transform(x_macro)
            error_ae_z = adaled.to_numpy(adaled.nn.loss.mse_losses(z_macro2, z_macro))

            x_macro2 = transformer.inverse_transform(z_macro2).cpu()
            error_abs2 = weighted_mse_losses(
                    computed=x_macro2, expected=x_macro, weight=None, relative=False)
            error_rel2 = weighted_mse_losses(
                    computed=x_macro2, expected=x_macro, weight=None, relative=True)

        out = adaled.TensorCollection({
            # x_micro vs Decoder(z_macro), absolute MSE.
            'macro_micro_x_abs': error_abs,
            # x_micro vs Decoder(z_macro), relative MSE.
            'macro_micro_x_rel': error_rel,
            # Encoder(x_micro) vs z_macro, MSE.
            'macro_micro_z': error_z,
            # Encoder(Decoder(z_macro)) vs z_macro, MSE.
            'macro_ae_z': error_ae_z,
            # Decoder(Encoder(Decoder(z_macro))) vs Decoder(z_macro), absolute MSE.
            'macro_ae_x_abs': error_abs2,
            # Decoder(Encoder(Decoder(z_macro))) vs Decoder(z_macro), relative MSE.
            'macro_ae_x_rel': error_rel2,
        }).numpy()
        return out


def main(argv=None):
    main = adaled.Main()
    config = main.parse_and_process(argv, Config())
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

    F_func = config.Fd.make_func(config.led.max_steps, config.rd.dt_macro)

    rnns = [adaled.RNN(config.rnn) for _ in range(config.ensemble_size)]
    adaled.print_model_info(rnns[0])
    macro = adaled.ProbabilisticPropagatorEnsemble(rnns)
    macro_trainer = macro.make_trainer(config.ensemble_training)

    autoencoder = config.encoder.make_autoencoder()
    autoencoder.model.encoder.describe()
    autoencoder.model.decoder.describe()
    if config.scale != 1.0:
        scaling = adaled.Scaling(min=-config.scale, max=+config.scale)
        print(f"Using {scaling}.")
        encoder = torch.nn.Sequential(scaling.to_torch(), autoencoder.model.encoder)
        decoder = torch.nn.Sequential(autoencoder.model.decoder, scaling.inversed().to_torch())
        autoencoder = adaled.AutoencoderTransformer(encoder, decoder)
    autoencoder_trainer = adaled.AutoencoderTrainer(
            autoencoder.model,
            **config.ae_training.make(autoencoder.model.parameters()),
            loss=adaled.MSELosses())

    trainer = config.led_trainer.make(
            macro_trainer=macro_trainer, transformer=autoencoder,
            transformer_trainer=autoencoder_trainer)

    trajectory_datasets = data_collections.CappedFixedLengthTrajectoryDatasetCollection(
            config.train_capacity, config.valid_capacity, config.trajectory_length)

    led = adaled.AdaLED(
            macro=macro,
            trainer=trainer,
            transformer=autoencoder,
            config=config.led,
            criteria=config.criteria.create(),
            datasets=trajectory_datasets)

    micro = ReactionDiffusionSolver(config.rd)

    # TODO: Generator for F is not that practical. Replace with something else.
    F_generator = adaled.utils.misc.function_to_generator(F_func, config.rd.dt_macro)
    generator = led.make_generator(micro, F_generator)
    recorder = Recorder(config, generator)
    led.run(generator, recorder)
