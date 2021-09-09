#!/usr/bin/env python3

from adaled.tools.hyperparams import SlurmHyperParameterStudy
import adaled

import numpy as np

import os

DIR = os.path.dirname(os.path.abspath(__file__))

_SLURM_TEMPLATE = '''\
#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition=normal
#SBATCH --constraint=gpu

{venv}
module swap -s PrgEnv-cray PrgEnv-gnu 2> /dev/null
module load daint-gpu
module rm cudatoolkit cray-mpich cray-hdf5-*
module load cray-mpich
module load GREASY
module load cray-python  # Must be after greasy.

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS={cpus_per_task}

greasy {greasy_file}
'''

class Study(SlurmHyperParameterStudy):
    RUNSCRIPT_PATH = os.path.join(DIR, 'run.py')
    PATH_PREFIX = 'run'
    SLURM_TEMPLATE = _SLURM_TEMPLATE
    DEFAULT_KWARGS = {
        '--ntasks-per-node': 12,
        '--cpus-per-task': 1,
        '--nodes': 16,
        '--combinations': 1536,
        '--prepare-folders': 1,
        # Sort such that slow examples are first.
        # has_sigma2==1 is slower htan has_sigma2==0 (sort DESC)
        # smaller batches are slower (sort ASC)
        '--sort': '-network.has_sigma2,+led_trainer.macro_batch_size',
    }

    def get_param_space(self, args):
        import scipy.stats as stats
        space = {
            '--seed': stats.randint(0, 10000),
            'network.rnn_hidden_size': [8, 16, 32],
            'network.rnn_num_layers': [1, 2, 3, 4],
            'network.append_F': [0, 1],
            'network.has_sigma2': [0, 1],

            # hyper-01
            # 'num_parallel_sim': [1, 8],
            # 'ensemble_training.lr': stats.loguniform(0.0002, 0.02),
            # 'led_trainer.macro_batch_size': [8, 16, 32],
            # 'led_trainer.trajectories_sampling_policy.type': ['random', 'preferential'],

            # hyper-02/03
            'num_parallel_sim': [1],
            # 'ensemble_training.lr': stats.loguniform(0.0002, 0.01),
            # 'led_trainer.trajectories_count_policy.fraction': [0.1],
            'criteria.num_relaxation_steps': [0, 10],
            # 'criteria.max_cmp_error': [0.01, 0.005],
            # 'criteria.max_uncertainty': [0.01, 0.005],

            # hyper-04
            # No batch of 8, it is too slow.
            'led_trainer.macro_batch_size': [16, 32, 64],
            'led_trainer.trajectories_count_policy.per_timestep': [2.0],
            'criteria.max_cmp_error': [0.01],
            'criteria.max_uncertainty': [0.01],
            'ensemble_training.lr': stats.loguniform(0.0002, 0.02),
            'nll_weight': [1e-4, 1e-5, 1e-6],
            'adversarial_eps': [0.0, 0.0, 0.0, 0.0, 0.0001, 0.001, 0.01, 0.03],
            'adversarial_mse_weight:json': [0.0, 0.01, 0.1, 1.0],

            'led.dump_dataset.every': [50],
            'led.dump_macro.every': [50],
            'led.dump_transformer.every': [50],

            # Record everything, but in single precision and with fewer files.
            # Should take on average ~300MB per run.
            'recorder.start_every': [50000],
            'recorder.num_steps': [50000],
            'recorder.posttransform': ['float32'],
            'dataset_histograms': [{}],
        }
        return space

    def generate_sample_05(
            self, kwargs: dict, random_state: np.random.RandomState,
            sample_idx: int, num_samples: int, **kw):
        """Test MSE vs NLL."""
        weights = [(1.0, 0.0), (1.0, 1e-30), (1.0, 1e-5), (1.0, 1e-3), (1.0, 1.0), (0.0, 1.0), (0.0, 1e-3), (0.0, 1e-5)]
        mse_weight, nll_weight = weights[len(weights) * sample_idx // num_samples]
        kwargs['mse_weight'] = mse_weight
        kwargs['nll_weight'] = nll_weight
        kwargs['network.has_sigma2'] = int(nll_weight > 0)
        return kwargs

    def generate_sample_06(
            self, kwargs: dict, random_state: np.random.RandomState,
            sample_idx: int, num_samples: int, **kw):
        """Hyperparam study with values different than the production runs."""
        kwargs['mu.values'] = [1.5, 2.5]
        kwargs['mu.T'] = 2500
        # Keep max_cmp_error and max_uncertainty to 0.01 instead of 0.005.
        # kwargs['adversarial_eps'] = 0.0
        # kwargs['adversarial_mse_weight:json'] = 0.0
        kwargs['mse_weight'] = 1.0
        kwargs['nll_weight'] = 1.0
        kwargs['network.has_sigma2'] = 1
        kwargs['network.detach_sigma2'] = 1
        kwargs['criteria.num_relaxation_steps'] = 0
        kwargs['led.max_steps'] = 200010
        return kwargs


    generate_sample = generate_sample_06

    def format_path_suffix(self, i: int, kwargs: dict):
        kwargs = {key.replace('.', '_'): value for key, value in kwargs.items()}
        fmt = '-{i:04d}-p{num_parallel_sim}' \
              '-h{network_rnn_num_layers}x{network_rnn_hidden_size}' \
              '-sigma{network_has_sigma2}-F{network_append_F}' \
              '-lr{ensemble_training_lr:.2e}'
        return fmt.format(i=i, **kwargs)


if __name__ == '__main__':
    study = Study()
    study.main()
