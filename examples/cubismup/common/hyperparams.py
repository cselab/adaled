from adaled.tools.hyperparams import SlurmHyperParameterStudy
import adaled

import os
import re

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
export MPICH_MAX_THREAD_SAFETY=multiple  # For Cubism.

INDEX="${{ADALED_LAUNCH_INDEX:-0}}"  # For multi-job runs.
CMD="srun --wait=0 --kill-on-bad-exit --nodes={nodes} --ntasks={ntasks} --ntasks-per-node={ntasks_per_node}"
CMD+=" --output=stdout-${{INDEX}}-%t.txt"
CMD+=" --error=stderr-${{INDEX}}-%t.txt"

if [ {ntasks_per_node} == 2 ]; then
    # Force a cpu binding mask to distribute client and server threads on
    # different virtual cores. 2D benchmarks showed that fff000,000fff with 12
    # threads is better, but 3D benchmarks now show that c00c00,3ff3ff with 10
    # threads is better.
    CMD+=' --cpu-bind=mask_cpu:{cpu_bind}'
fi

ADALED_COMMAND_PREFIX="${{CMD}}" ./exec_nth.sh $SLURM_ARRAY_TASK_ID
'''

class CUPSlurmHyperParameterStudy(SlurmHyperParameterStudy):
    SLURM_TEMPLATE = _SLURM_TEMPLATE
    PATH_PREFIX = 'run'
    DEFAULT_KWARGS = {
        # Server and client on the same rank.
        # '--ntasks-per-node': 2,  # 1 server & 1 client.
        # '--nodes': 1,
        # Server and client separately.
        '--ntasks-per-node': 1,
        '--nodes': 2,

        '--cpus-per-task': 12,
        '--combinations': 10,
        '--pipe-stdio': 0,
    }
    DEFAULT_TEMPLATE_ARGS = {
        # 'omp_num_threads': 10,
        # 'cpu_bind': 'c00c00,3ff3ff'   # For 2D.
        # 'omp_num_threads': 12,
        # 'cpu_bind': 'fff000,000fff',  # For 3D.
    }

    def postprocess_diagnostics(self, output_dir, *args, **kwargs) -> dict:
        post = super().postprocess_diagnostics(output_dir, *args, **kwargs)
        extra = adaled.load(os.path.join(output_dir, 'metadata.json'))
        post.update(extra)

        # Include data from the expensive postprocessing run.
        try:
            path = os.path.join(output_dir, 'postprocessed-000.pt')
            postprocessed = adaled.load(path)
        except FileNotFoundError:
            print(f"{path} not found, did you run ./postprocess.sh?")
        else:
            from .postprocess import compute_cumulative_stats
            post.update(compute_cumulative_stats(postprocessed))

        # Clean up annoying network validation losses (not the final cmp/validation error).
        remove_re = re.compile(r'.*losses\..*_valid\..*')
        post = {k: v for k, v in post.items() if not remove_re.match(k)}
        return post


if __name__ == '__main__':
    study = Study()
    study.main()
