#!/usr/bin/env python3

from examples.cubismup.common.hyperparams import CUPSlurmHyperParameterStudy
from adaled.led import AdaLEDStage as Stage
import adaled

import numpy as np

from typing import Sequence
import os

# Model comparison 6 study results:
#   - 2 resolutions better than 1 resolution with 10-15% higher macro
#     utilization, for the same velocity relative MSE and force MSE (averaged
#     over macro-only steps).
#   - 224x224 vs 256x256 are more or less the same, sometimes one is better,
#     sometimes the other.
#   - Selected optimal parameters (rounded):
#       autoencoder.training.lr=0.00015
#       macro.training.lr=0.001
#       led.criteria.max_cmp_error=0.014
#       led.criteria.max_uncertainty=0.00067
#       'autoencoder.encoders.0.conv_layers_kernel_sizes:json=[5,5,5,5,3]'
#       'autoencoder.encoders.0.conv_layers_channels:json=[16,16,16,16,16]'
#       autoencoder.encoders.0.latent_state_dim=12
#       cyl_forces_ae_scaling=6.0


class Study(CUPSlurmHyperParameterStudy):
    RUNSCRIPT_PATH = 'python3 -m examples.cubismup.cyl2d.run'
    HYPERPARAM_SCRIPT_PATH = 'python3 -m examples.cubismup.cyl2d.hyperparams'
    DEFAULT_KWARGS = {
        # Server and client on the same rank.
        # '--ntasks-per-node': 2,  # 1 server & 1 client.
        # '--nodes': 1,
        # Server and client separately.
        '--ntasks-per-node': 1,
        '--nodes': 2,
        '--cpus-per-task': 12,
        '--combinations': 128,
        '--pipe-stdio': 0,
    }
    DEFAULT_TEMPLATE_ARGS = {
        'omp_num_threads': 12,
        'cpu_bind': 'fff000,000fff',
    }

    def get_param_space(self, args):
        import scipy.stats as stats
        space = {
            '--seed': stats.randint(0, 10000),
            'micro.dt_macro': [0.005],  # ~120 steps/period for Re=500, ~60 for Re=1000
            'Re.T': [50.0],    # change every 10000 steps
            'led.client.max_steps': [50102],  # ~1 day
            # 20 mins should be enough to finish a cycle even with always_run_micro=1 and max_macro_steps=500.
            'led.server.max_seconds': [24 * 3600 - 1200],

            'micro.cells': [[1024, 512]],
            'micro.double_precision': [0, 0, 0, 1],
            'macro.rnn.rnn_hidden_size': [32],
            'macro.rnn.rnn_num_layers': [2],
            'macro.rnn.has_sigma2': [1],
            'macro.rnn.append_F': [1],

            'autoencoder.encoders.0.latent_state_dim': [2, 3, 4, 6, 8],
            'autoencoder.training.lr': stats.loguniform(0.00001, 0.001),
            'macro.training.lr': stats.loguniform(0.00002, 0.01),
            # '--ranks-per-simulation': [2],  # Must be equal to slurm --nodes!
            'led.led_trainer.macro_batch_size': [8, 16],           # Local.
            'led.led_trainer.transformer_batch_size': [4, 8, 16],  # Local.

            'led.criteria.max_cmp_error': [1e-9],
            'led.criteria.max_uncertainty': [1e-9],

            'led.server.dump_dataset.every': [50],
            'led.server.dump_macro.every': [50],
            'led.server.dump_transformer.every': [50],
            'led.server.dump_trainers.every': [50],

            'led.recorder.start_every': [1000],  # Have fewer files.
            'led.recorder.num_steps': [1000],
            'led.recorder.every': [1],
            'led.recorder.x_every': [100],
            'led.recorder.z_every': [1],

            # Histograms greatly slow down the server, disable.
            'led.dataset_histograms': [{}],
        }
        return space

    def generate_sample1(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        def sample(choices):
            return choices[random.choice(len(choices))]

        random = random_state
        kwargs['autoencoder.vorticity_in_encoder'] = random.choice(2)

        num_layers = 4 + random.choice(2)
        if num_layers == 4:
            choices = [[10, 12, 14, 16], [12, 14, 16, 18], [14, 16, 18, 20]]
        else:
            choices = [[10, 12, 14, 16, 18], [12, 14, 16, 18, 20], [14, 16, 18, 20, 22]]
        kwargs['autoencoder.encoders.0.conv_layers_channels'] = sample(choices)
        kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5] * num_layers
        kwargs['autoencoder.encoders.0.conv_layers_strides'] = [1] * num_layers
        kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [2] * num_layers
        kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [2] * num_layers

        kwargs['micro.vorticity_in_state'] = \
                kwargs['autoencoder.vorticity_in_encoder'] and random.choice(2)

        kwargs['micro.extra_cubism_argv:extend'] = sample([[], ['-poissonSolver', 'cuda_iterative']])
        return kwargs

    def generate_sample2(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs['Re.T'] = 40.0  # Every 8000 time steps.
        kwargs['led.client.max_steps'] = 40000   # ~22h
        kwargs['led.criteria.max_cmp_error'] = sample([1e-4, 3e-4, 5e-4])
        kwargs['led.criteria.max_uncertainty'] = sample([3e-4, 1e-3])
        kwargs['led.criteria.num_relaxation_steps'] = sample([0, 5, 15, 140])
        kwargs['led.led_trainer.macro_batch_size'] = 8
        kwargs['led.led_trainer.transformer_batch_size'] = sample([8, 16])  # Local.
        kwargs['autoencoder.encoders.0.conv_layers_channels'] = \
                sample([[10, 12, 14, 16], [12, 14, 16, 18]])
        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([8, 12, 16])
        kwargs['autoencoder.vorticity_in_encoder'] = 0
        kwargs['autoencoder.training.lr'] = 3e-4
        kwargs['macro.training.lr'] = 1e-3
        kwargs['micro.double_precision'] = sample([0, 1])
        kwargs['micro.extra_cubism_argv:extend'] = ['-poissonSolver', 'cuda_iterative']
        kwargs['micro.vorticity_in_state'] = 0
        return kwargs

    def generate_sample3(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        from scipy.stats import loguniform

        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs['Re.T'] = 40.0  # Every 8000 time steps.
        kwargs['led.client.max_steps'] = 40000   # ~22h
        kwargs['led.criteria.max_cmp_error'] = sample([1e-5, 1e-4, 1e-3, 1e-2])
        kwargs['led.criteria.max_uncertainty'] = sample([1e-4, 1e-3])
        kwargs['led.criteria.num_relaxation_steps'] = 0
        kwargs['led.led_trainer.macro_batch_size'] = 8
        kwargs['led.led_trainer.transformer_batch_size'] = sample([8, 16])  # Local.
        kwargs['autoencoder.encoders.0.conv_layers_channels'] = \
                sample([[10, 12, 14, 16], [12, 14, 16, 18], [14, 16, 18, 20]])
        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12])
        kwargs['autoencoder.loss.vorticity_weight'] = loguniform(0.0001, 1.0).rvs(random_state=random_state)
        kwargs['autoencoder.loss.relative_loss'] = 1
        kwargs['autoencoder.training.lr'] = loguniform(0.00005, 0.002).rvs(random_state=random_state)
        kwargs['autoencoder.vorticity_in_encoder'] = 0
        kwargs['macro.training.lr'] = 1e-3
        kwargs['micro.double_precision'] = 0
        kwargs['micro.extra_cubism_argv:extend'] = ['-poissonSolver', 'cuda_iterative']
        kwargs['micro.vorticity_in_state'] = 0
        return kwargs

    def generate_sample4(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        from scipy.stats import loguniform

        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs['Re.T'] = 40.0  # Every 8000 time steps.
        kwargs['led.client.max_steps'] = 32000  # NOTE: The last Re is 500, not 1000!
        kwargs['led.criteria.max_cmp_error'] = 1e-5
        kwargs['led.criteria.max_uncertainty'] = 1e-4
        kwargs['led.criteria.num_relaxation_steps'] = 0
        kwargs['led.led_trainer.macro_batch_size'] = 8
        kwargs['led.led_trainer.transformer_batch_size'] = sample([8, 16])  # Local.
        kwargs['autoencoder.loss.vorticity_weight'] = loguniform(0.0001, 1.0).rvs(random_state=random_state)
        kwargs['autoencoder.loss.relative_loss'] = 1
        kwargs['autoencoder.training.lr'] = loguniform(0.00005, 0.002).rvs(random_state=random_state)
        kwargs['autoencoder.vorticity_in_encoder'] = 0
        kwargs['macro.training.lr'] = 1e-3
        kwargs['micro.double_precision'] = 0
        kwargs['micro.extra_cubism_argv:extend'] = ['-poissonSolver', 'cuda_iterative']
        kwargs['micro.vorticity_in_state'] = 0
        kwargs['micro.multiresolution.1.center'] = [0.2, 0.25]
        kwargs['micro.multiresolution.1.size'] = [224, 224]
        kwargs['micro.multiresolution.0.alpha_margin_cells:json'] = 22.0
        kwargs['micro.multiresolution.1.alpha_margin_cells:json'] = 22.0
        kwargs['micro.multiresolution.0.alpha_sigma_cells:json'] = 3.0
        kwargs['micro.multiresolution.1.alpha_sigma_cells:json'] = 3.0
        kwargs['autoencoder.loss.layer_margin_cells'] = 10

        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12])
        num_layers = 4 + random_state.choice(2)
        if num_layers == 4:
            choices = [[16, 16, 16, 16], [20, 20, 20, 20], [24, 24, 24, 24]]
        else:
            choices = [[16, 16, 16, 16, 16], [20, 20, 20, 20, 20], [24, 24, 24, 24, 24]]
            kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5, 5, 5, 5, 3]
            kwargs['autoencoder.encoders.0.conv_layers_strides'] = [1, 1, 1, 1, 1]
            kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [2, 2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [2, 2, 2, 2, 2]
        kwargs['autoencoder.encoders.0.conv_layers_channels'] = sample(choices)
        return kwargs

    def generate_sample5_model_cmp(
            self, kwargs: dict, random_state: np.random.RandomState,
            sample_idx: int, num_samples: int, **kw):
        """Compare no multiresolution vs multiresolution with either 256x256 or
        224x224. Compare different max_uncertainty and max_cmp_error. Compare
        different AE and latent sizes."""
        from scipy.stats import loguniform

        del kwargs['micro.double_precision']

        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs['Re.T'] = 40.0  # Every 8000 time steps.
        kwargs['led.client.max_steps'] = 80000
        kwargs['led.criteria.max_cmp_error'] = loguniform(0.001, 0.1).rvs(random_state=random_state)
        kwargs['led.criteria.max_uncertainty'] = loguniform(0.00001, 0.1).rvs(random_state=random_state)
        kwargs['led.criteria.num_relaxation_steps'] = 0
        kwargs['led.led_trainer.macro_batch_size'] = 8
        kwargs['led.led_trainer.transformer_batch_size'] = 8  # Local.
        kwargs['autoencoder.training.lr'] = 0.0003
        kwargs['macro.training.lr'] = 1e-3
        arch = 3 * sample_idx // num_samples  # (1 mr, 2 mr 256x256, 2 mr 224x224)
        if arch == 0:
            kwargs['micro.multiresolution.1:delete'] = ''
            kwargs['micro.multiresolution.0.stride'] = 1
            kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([8, 16, 24])
            num_layers = 5 + random_state.choice(2)
        else:
            kwargs['micro.multiresolution.1.size'] = [224, 224] if arch == 1 else [256, 256]
            kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12])
            # Record full resolution to be able to compute the true error,
            # without the downscaling. For arch == 0, states are stored in full
            # resolution anyway.
            kwargs['micro.record_full_resolution'] = 1
            num_layers = 4 + random_state.choice(2)
        channels_per_layer = sample([16, 20, 24])
        kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5] * (num_layers - 1) + [3]
        kwargs['autoencoder.encoders.0.conv_layers_channels'] = [channels_per_layer] * num_layers
        kwargs['led.recorder.x_every'] = 1  # Warning: records everything, generates ~320-560GB per run!
        kwargs['micro.extra_cubism_argv:extend'] = ['-poissonSolver', 'cuda_iterative']
        return kwargs

    def generate_sample6_model_cmp(
            self, kwargs: dict, random_state: np.random.RandomState, **kw):
        from scipy.stats import loguniform
        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs = self.generate_sample5_model_cmp(kwargs, random_state, **kw)
        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12, 16, 24])
        kwargs['autoencoder.training.lr'] = loguniform(0.0001, 0.001).rvs(random_state=random_state)
        # kwargs['autoencoder.loss.derivatives_weight'] = 0.03
        # kwargs['autoencoder.loss.vorticity_weight'] = 0.0
        kwargs['macro.training.lr'] = loguniform(0.0003, 0.003).rvs(random_state=random_state)
        kwargs['cyl_forces_ae_scaling'] = loguniform(0.03, 30.0).rvs(random_state=random_state)
        kwargs['micro.forces_in_state'] = 'total'
        kwargs['micro.record_full_resolution'] = 0
        kwargs['led.client.always_run_micro'] = 1  # Server will automatically compensate for this.
        kwargs['led.recorder.x_every'] = 100       # No need to record everything since always_run_micro=1.
        kwargs['led.server.restart'] = 1           # Restart if dump files found.
        kwargs['led.server.compensate_for_stats_overhead'] = 1  # Default, but let it be explicit.
        return kwargs

    def generate_sample7_model_cmp(
            self, kwargs: dict, random_state: np.random.RandomState, **kw):
        """Re-run of the architecture comparison. This time with a different
        setup than for the production run."""
        from scipy.stats import loguniform
        kwargs = self.generate_sample6_model_cmp(kwargs, random_state, **kw)
        kwargs['Re.T'] = 25.0                   # Every 5000 time steps.
        kwargs['Re.values'] = [600, 750, 900]   # Make it different than 500-1000.
        kwargs['led.client.max_steps'] = 60000  # Make it different than 500-1000.

        # Narrowing the range of max_uncertainty was a bad idea, these points
        # are part of the pareto front.
        # kwargs['led.criteria.max_uncertainty'] = loguniform(0.0001, 0.1).rvs(random_state=random_state)
        kwargs['led.criteria.max_uncertainty'] = loguniform(0.00001, 0.0001).rvs(random_state=random_state)
        # kwargs['led.criteria.max_uncertainty'] = loguniform(0.00001, 0.1).rvs(random_state=random_state)
        return kwargs

    @staticmethod
    def update_sample7_model_cmp(kwargs: dict, random_state: np.random.RandomState):
        def update(key: str, low: float, high: float):
            factor = random_state.lognormal(0.0, 0.1)
            kwargs[key] = max(low, min(high, kwargs[key] * factor))
            pass
        update_log('led.criteria.max_uncertainty', 0.00001, 0.1)
        update_log('autoencoder.training.lr', 0.0001, 0.001)
        update_log('macro.training.lr', 0.0003, 0.003)
        update_log('cyl_forces_ae_scaling', 0.03, 30.0)

    generate_sample = generate_sample7_model_cmp

    def format_path_suffix(self, i: int, kwargs: dict):
        if 'micro.multiresolution.1:delete' in kwargs:
            mrlen = '1'
        else:
            mrlen = '2-{}'.format(kwargs['micro.multiresolution.1.size'][0])

        kwargs = {
            key.replace('autoencoder.encoders.0', 'AE').replace('.', '__'): value
            for key, value in kwargs.items()
        }
        fmt = '-{i:04d}-mr{mrlen}-z{AE__latent_state_dim}'
        return fmt.format(i=i, mrlen=mrlen, **kwargs)

    def postprocess_diagnostics(
            self,
            output_dir: str,
            *args,
            diagnostics: Sequence[adaled.TensorCollection],
            **kwargs) -> dict:
        post = super().postprocess_diagnostics(
                output_dir, *args, diagnostics=diagnostics, **kwargs)

        # Load runtime (online) postprocessed data.
        # TODO: Make this part of common, and somehow make an abstract load* function.
        from .postprocess_utils import load_and_extend_runtime_postprocess_data
        try:
            path = os.path.join(output_dir, 'postprocessed-runtime-000.pt')
            assert len(diagnostics) == 1, len(diagnostics)
            diag = adaled.AdaLEDDiagnostics(per_cycle_stats=adaled.DynamicArray(diagnostics[0]))
            runtime = load_and_extend_runtime_postprocess_data(path, diag)
            runtime = runtime[1:]  # Skip first step, it has some nans.
            is_macro = runtime['metadata', 'stage'] == Stage.MACRO
            post['runtime.cmp_error.v'] = \
                    runtime['cmp_error', 'v'][is_macro].sum() / len(is_macro)
            post['runtime.__qoi_error_l2.cyl_forces'] = \
                    runtime['__qoi_error_l2', 'cyl_forces'][is_macro].sum() / len(is_macro)
        except FileNotFoundError:
            import warnings
            warnings.warn(f"{path} not found, did you run ./postprocess.sh?")

        return post


if __name__ == '__main__':
    study = Study()
    study.main()
