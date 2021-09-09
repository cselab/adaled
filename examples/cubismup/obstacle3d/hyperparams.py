from ..common.hyperparams import CUPSlurmHyperParameterStudy
import os
import adaled

import numpy as np

import os

DIR = os.path.dirname(os.path.abspath(__file__))


# Conclusions from the hyperparameter studies.
#
# Study #1 (256 x 128 x 128):
# Parameters:
#   - double vs float precision
#   - transformer batch size 4 vs 2
#   - different number of channels
# Results:
#   - double precision makes training 3-4x time slower than single precision!
#   - double precision makes clients ~5-10% slower than single precision
#   - with transformer batch size of 4, 20% more samples are trained on,
#     compared to batch size of 2
#   - number of channels doesn't really affect the training speed
# Conclusions:
#   - prior to macro training, we could use a larger batch size for the
#     transformer, since this part does not need memory for gradients
#


class Study(CUPSlurmHyperParameterStudy):
    RUNSCRIPT_PATH = 'python3 -m examples.cubismup.obstacle3d.run'
    HYPERPARAM_SCRIPT_PATH = 'python3 -m examples.cubismup.obstacle3d.hyperparams'
    DEFAULT_TEMPLATE_ARGS = {
        'omp_num_threads': 10,
        'cpu_bind': 'c00c00,3ff3ff'
    }

    def get_param_space(self, args):
        import scipy.stats as stats
        space = {
            '--seed': stats.randint(1000, 9999),
            'Re.values': [[300.0]],
            'Re.T': [1000000.0],    # never change
            'led.client.max_steps': [10000],  # ~7h
            'micro.dt_macro': [0.040],
            'micro.cells': [[256, 128, 128]],

            # hyper-02
            'autoencoder.encoders.0.conv_layers_kernel_sizes': [[5, 5, 5, 5]],
            'autoencoder.encoders.0.conv_layers_channels':
                [[8, 10, 12, 14], [10, 12, 14, 16], [12, 14, 16, 18]],
            'autoencoder.encoders.0.conv_layers_strides': [[1, 1, 1, 1]],
            'autoencoder.encoders.0.pool_kernel_strides': [[2, 2, 2, 2]],
            'autoencoder.encoders.0.pool_kernel_sizes': 'autoencoder.encoders.0.pool_kernel_strides',
            'autoencoder.encoders.0.latent_state_dim': [4, 6],
            'autoencoder.encoders.0.batch_norm': [0, 1],
            'autoencoder.training.lr': stats.loguniform(0.00001, 0.001),
            'autoencoder.loss.vorticity_weight': [1e-5, 1e-4],
            'macro.training.lr': stats.loguniform(0.00002, 0.01),
            'micro.double_precision': [0, 1],
            'led.led_trainer.macro_batch_size': [32],
            'led.led_trainer.transformer_batch_size': [2, 4],  # Local.

            'dataset.train_capacity': [128],
            'dataset.valid_capacity': [16],
            'led.server.dump_dataset.every': [20],
            'led.server.dump_macro.every': [20],
            'led.server.dump_transformer.every': [20],
            'led.server.dump_trainers.every': [20],

            # Histograms slow down the server, disable.
            'led.dataset_histograms': [{}],
        }
        return space

    def generate_sample3(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        def sample(choices):
            return choices[random_state.choice(len(choices))]

        kwargs['autoencoder.encoders.0.conv_layers_channels'] = \
                sample([[8, 10, 12, 14], [10, 12, 14, 16], [12, 14, 16, 18], [14, 16, 18, 20]])
        kwargs['led.client.max_steps'] = 30000  # ~20 h
        kwargs['led.led_trainer.macro_batch_size'] = 8        # Local.
        kwargs['led.led_trainer.transformer_batch_size'] = sample([4, 8])  # Local.
        kwargs['led.recorder.x_every'] = 100
        kwargs['autoencoder.encoders.0.batch_norm'] = 0
        kwargs['autoencoder.encoders.0.conv_layers_strides'] = [2, 1, 1, 1]
        kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [1, 2, 2, 2]
        kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [1, 2, 2, 2]
        kwargs['autoencoder.vorticity_in_encoder'] = 0
        kwargs['micro.double_precision'] = 0
        kwargs['micro.vorticity_in_state'] = 0

        return kwargs

    def generate_sample4(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        def sample(choices):
            return choices[random_state.choice(len(choices))]
        kwargs = self.generate_sample3(kwargs, random_state)

        if random_state.choice(2) == 0:
            kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5, 5, 5, 3]
            kwargs['autoencoder.encoders.0.conv_layers_strides'] = [2, 1, 1, 1]
            kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [1, 2, 2, 2]
            kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [1, 2, 2, 2]
            kwargs['autoencoder.encoders.0.conv_layers_channels'] = sample([[12, 12, 12, 12], [16, 16, 16, 16], [20, 20, 20, 20]])
        else:
            kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5, 5, 5, 3, 3]
            kwargs['autoencoder.encoders.0.conv_layers_strides'] = [2, 1, 1, 1, 1]
            kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [1, 2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [1, 2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.conv_layers_channels'] = sample([[12, 12, 12, 12, 12], [16, 16, 16, 16, 16], [20, 20, 20, 20, 20]])

        kwargs['autoencoder.encoders.0.batch_norm'] = 1
        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12])
        kwargs['autoencoder.loss.vorticity_weight'] = 1e-4

        return kwargs

    def generate_sample5(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        def sample(choices):
            return choices[random_state.choice(len(choices))]
        kwargs = self.generate_sample3(kwargs, random_state)

        if random_state.choice(2) == 0:
            kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5, 5, 5, 3]
            kwargs['autoencoder.encoders.0.conv_layers_strides'] = [1, 1, 1, 1]
            kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.conv_layers_channels'] = \
                    sample([[12, 12, 12, 12], [16, 16, 16, 16], [20, 20, 20, 20], [24, 24, 24, 24]])
        else:
            kwargs['autoencoder.encoders.0.conv_layers_kernel_sizes'] = [5, 5, 5, 3, 3]
            kwargs['autoencoder.encoders.0.conv_layers_strides'] = [1, 1, 1, 1, 1]
            kwargs['autoencoder.encoders.0.pool_kernel_sizes'] = [2, 2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.pool_kernel_strides'] = [2, 2, 2, 2, 2]
            kwargs['autoencoder.encoders.0.conv_layers_channels'] = \
                    sample([[12, 12, 12, 12, 12], [16, 16, 16, 16, 16], [20, 20, 20, 20, 20], [24, 24, 24, 24, 24]])

        kwargs['autoencoder.scaling.potential_margin'] = sample([100, 200, 500, 1000, 1500])
        kwargs['autoencoder.encoders.0.batch_norm'] = sample([0, 1])
        kwargs['autoencoder.encoders.0.latent_state_dim'] = sample([4, 8, 12])
        kwargs['autoencoder.loss.vorticity_weight'] = 1e-4

        kwargs['micro.multiresolution.1.center'] = [0.2, 0.25, 0.25]
        kwargs['micro.multiresolution.0.alpha_margin_cells:json'] = 5.0
        kwargs['micro.multiresolution.0.alpha_sigma_cells:json'] = 1.0
        kwargs['micro.multiresolution.1.alpha_margin_cells:json'] = 5.0
        kwargs['micro.multiresolution.1.alpha_sigma_cells:json'] = 1.0

        return kwargs

    generate_sample = generate_sample5

    def format_path_suffix(self, i: int, kwargs: dict):
        float_ = 'f64' if kwargs['micro.double_precision'] else 'f32'
        kwargs = {
            key.replace('autoencoder.encoders.0', 'AE').replace('.', '__'): value
            for key, value in kwargs.items()
        }
        fmt = '-{i:04d}-{float_}-b{AE__batch_norm}-z{AE__latent_state_dim}' \
              '-aelr{autoencoder__training__lr:.2e}-s{--seed:05d}'
        return fmt.format(i=i, float_=float_, **kwargs)


if __name__ == '__main__':
    study = Study()
    study.main()
