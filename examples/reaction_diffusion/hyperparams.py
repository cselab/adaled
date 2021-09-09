#!/usr/bin/env python

import os
import re

import numpy as np

from adaled.tools.hyperparams import SlurmHyperParameterStudy
import adaled

DIR = os.path.dirname(os.path.abspath(__file__))

# For Piz Daint.
_SLURM_TEMPLATE = '''\
#!/bin/bash -l
#SBATCH --array=0-{combinations_minus_1}
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition=normal
#SBATCH --constraint=gpu

{venv}
export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS={omp_num_threads}

INDEX="${{ADALED_LAUNCH_INDEX:-0}}"  # For multi-job runs.
CMD="srun --wait=0 --kill-on-bad-exit --nodes={nodes} --ntasks={ntasks} --ntasks-per-node={ntasks_per_node}"
CMD+=" --output=stdout-${{INDEX}}-%t.txt"
CMD+=" --error=stderr-${{INDEX}}-%t.txt"

ADALED_COMMAND_PREFIX="${{CMD}}" ./exec_nth.sh $SLURM_ARRAY_TASK_ID
'''

class Study(SlurmHyperParameterStudy):
    SLURM_TEMPLATE = _SLURM_TEMPLATE
    PATH_PREFIX = 'run'
    DEFAULT_KWARGS = {
        '--ntasks-per-node': 1,
        '--nodes': 1,
        '--cpus-per-task': 12,
        '--combinations': 10,
        '--pipe-stdio': 0,
    }
    RUNSCRIPT_PATH = 'python3 -m examples.reaction_diffusion.run'
    HYPERPARAM_SCRIPT_PATH = 'python3 -m examples.reaction_diffusion.hyperparams'
    DEFAULT_KWARGS = {
        '--ntasks-per-node': 1,
        '--nodes': 1,
        '--cpus-per-task': 12,
        '--combinations': 10,
        '--pipe-stdio': 0,
    }
    DEFAULT_TEMPLATE_ARGS = {
        'omp_num_threads': 12,
    }

    def get_param_space(self, args):
        import scipy.stats as stats
        space = {
            '--seed': stats.randint(0, 10000),
            'led.max_steps': [100004],
        }
        return space

    def generate_sample(self, kwargs: dict, random_state: np.random.RandomState, **kw):
        from scipy.stats import loguniform
        def sample(choices):
            return choices[random.choice(len(choices))]

        random = random_state
        kwargs['criteria.max_cmp_error'] = loguniform(0.0001, 0.003).rvs(random_state=random_state)
        kwargs['criteria.max_uncertainty'] = loguniform(0.001, 0.01).rvs(random_state=random_state)
        kwargs['encoder.conv_layers_channels'] = \
                sample([[10, 10, 10, 10], [12, 12, 12, 12], [14, 14, 14, 14]])
        kwargs['encoder.latent_state_dim'] = sample([2, 3, 4, 6, 8])
        kwargs['ae_training.lr'] = loguniform(0.0001, 0.01).rvs(random_state=random_state)
        kwargs['ensemble_training.lr'] = loguniform(0.001, 0.01).rvs(random_state=random_state)
        return kwargs



if __name__ == '__main__':
    study = Study()
    study.main()
