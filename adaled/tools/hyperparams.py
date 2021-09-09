import adaled

import numpy as np

from typing import Any, Dict, List, Optional, Sequence
import argparse
import json
import os
import re
import shlex
import sys

LAUNCH_LOCALLY_SH = '''\
#!/bin/bash

python3 -m adaled.tools.launch_greasy {greasy_file} "$@"
'''

PLOT_ALL_SH = '''\
#!/bin/bash

for dir in {path_prefix}*; do
    echo $dir
    (cd $dir && time ./plot.sh "$@")
done
'''

POSTPROCESS_SH = '''\
#!/bin/bash

{path} --postprocess --target-dir=. "$@"
'''

EXEC_NTH_SH = r'''#!/bin/bash

if [ $# != 1 ]; then
    echo "expected 1 argument, got $#"
    exit 1
fi

N0=$1
N=$(($1+1))
shift

ADALED_GREASY_FILE=${{ADALED_GREASY_FILE:-{greasy_file}}}

# Select N-th line starting with optional whitespace + '[@'.
LINE=$(grep '^\s*\[@' "$ADALED_GREASY_FILE" | sed -n "${{N}}p")
if [ -z "$LINE" ]; then
    echo "Line #$N0 (0-based) is empty or out of range."
    exit 1
fi

DIR="$(echo $LINE | sed 's/\[@ \(.*\) @\].*/\1/')"
mkdir -p "$DIR"
cd "$DIR"
eval $ADALED_COMMAND_PREFIX $(echo $LINE | sed 's/.*@\] \(.*\)/\1/') "$@"
'''

LAUNCH_SLURM_SH = '''\
#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: ./launch_slurm.sh <number of consecutive runs> [...]"
    exit 1
fi

COUNT=$1
shift

if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
    echo "First argument should be a number, got '$COUNT'."
    exit 1
fi

set -ex

for (( i=0; i<$COUNT; i++ )); do
    if [ $i == 0 ]; then
        DEPEND=
    else
        DEPEND=--depend=aftercorr:$JOBID
    fi
    JOBID=$(sbatch $DEPEND --export=ALL,ADALED_LAUNCH_INDEX=$i --parsable "$@" ./job_slurm.sh)
done
'''

def _time_format(value: str):
    if re.match(r'^\d\d:\d\d:\d\d$', value) is not None:
        return value
    else:
        raise ValueError(value)


def bash_gen_source_venv():
    """Return a bash command that loads the current environment."""
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        venv = 'source ' + os.path.join(venv, 'bin', 'activate') + '\n'
    else:
        venv = ''
    return venv


class HyperParameterStudy:
    RUNSCRIPT_PATH = None  # Required!
    HYPERPARAM_SCRIPT_PATH = None  # Automatically determined.
    DEFAULT_KWARGS = {}

    def __init__(self, default_kwargs: dict = {}):
        self.default_kwargs = self.DEFAULT_KWARGS.copy()
        self.default_kwargs.update(default_kwargs)

        self.greasy_file_path: Optional[str] = None
        self.task_names: Optional[List[str]] = None

    def create_parser(self, parser: Optional[argparse.ArgumentParser] = None):
        if not parser:
            parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        default = self.default_kwargs.get
        default_C = default('--combinations')
        add = parser.add_argument_group('sampling').add_argument
        add('-C', '--combinations', type=int, default=default_C,
            required=(default_C is None), help="number of hyperparameter sets")
        add('--seed', type=int, default=12345,
            help="seed used to generate random samples")
        add('extra_argv', nargs=argparse.REMAINDER,
            help="arguments appended to the runscript argument list")

        add = parser.add_argument_group('filesystem').add_argument
        add('--target-dir', type=str, required=True, help="target folder")
        add('--run-tasks-path', type=str,
            default=default('--run-tasks-path', 'tasks.run.txt'),
            help="Target greasy file for run tasks, relative to --target-dir.")
        add('--pipe-stdio', type=int,
            default=default('--pipe-stdio', True),
            help="whether or not to pipe stdout and stderr to a file")
        add('--prepare-folders', type=int,
            default=default('--prepare-folders', False),
            help="create folders in advance for each run")

        add = parser.add_argument_group('execution').add_argument
        add('--sort', type=str, default=default('--sort', ''),
            help="Comma-separated list of sorting keys for cases, used to "
                 "speed up greasy (slower cases should run first). Each key "
                 "should start with + for asc. and - for desc. sorting.")

        add = parser.add_argument_group('postprocessing').add_argument
        add('--postprocess', action='store_true', default=None)
        add('--post-avg-window', type=int, default=1000000000,
            help="averaging window when postprocessing diagnostics")
        add('--post-max-timesteps', type=int, default=0,
            help="restrict diagnostics to given number of time steps")
        add('--tmp-diag-rank', type=int,
            help="temporary flag used to customize which diagnostics is used, "
                 "for multi-client hyperparameter studies runs")
        return parser

    def main(self, argv: Optional[List[str]] = None):
        parser = self.create_parser()
        args = parser.parse_args(argv)
        if args.extra_argv and args.extra_argv[0] == '--':
            args.extra_argv = args.extra_argv[1:]
        if args.postprocess:
            self.postprocess(args)
        else:
            self.main_prepare(args)

    def main_prepare(self, args: Any):
        random = np.random.RandomState(args.seed)
        space = self.get_param_space(args)
        combinations = self.random_combinations(
                space, args.combinations, random, repeat=args.repeat_params)
        combinations = self.sort_combinations(combinations, args.sort)
        self.prepare_tasks(combinations, args.target_dir, args.run_tasks_path,
                           args.extra_argv, repeat_params=args.repeat_params,
                           pipe_stdio=args.pipe_stdio)
        if args.prepare_folders:
            self.prepare_folders(args.target_dir)
        self.prepare_shortcuts(args.target_dir)

    def get_param_space(self, args: Any):
        """Return a dictionary of keys and options.

        Example:
            def get_params_space(self, args):
                import scipy.stats as stats
                return {
                    'seed': stats.randint(0, 100000),
                    'example_str': ['this', 'or this'],
                    'example_log': stats.loguniform(1e-5, 1e-2),
                }
        """
        raise NotImplementedError(
                "get_param_space not implemented. Function docstring:\n\n"
                + self.get_param_space.__doc__)

    def generate_sample(
            self,
            kwargs: dict,
            random_state: np.random.RandomState,
            sample_idx: int,
            num_samples: int) -> Optional[dict]:
        """Generate a sample, given the keyword arguments produced from
        `get_param_space`. To reject a sample, return `None`.

        Arguments:
            kwargs: dictionary of config keys and their values
            random_state: random generator
            sample_idx: index of the current sample
            num_samples: total number of samples
        """
        return kwargs

    def format_path_suffix(self, idx: int, kwargs: Dict[str, Any]):
        """Generate a path suffix for the given set of configuration
        parameters."""
        return f'-{idx:06d}'

    def prepare_tasks(
            self,
            combinations: Sequence[Dict[str, Any]],
            target_dir: str,
            run_tasks_path: str,
            extra_argv: List[str],
            repeat_params: bool = False,
            pipe_stdio: bool = True):
        append = ' '.join(map(shlex.quote, extra_argv))
        if pipe_stdio:
            append += ' > stdout.txt 2> stderr.txt'
        run = []
        plot = []
        names = []
        for i, combination in enumerate(combinations):
            path = self.PATH_PREFIX + self.format_path_suffix(i, combination)
            if repeat_params:
                # Ensure unique paths when all runs have the same params.
                path += f'-{i:02d}'
            args = [self.format_argv(k, v) for k, v in combination.items()]
            args = ' '.join(map(shlex.quote, args))
            cmd = f'{self.RUNSCRIPT_PATH} --cwd=. {args} {append}'.strip()
            run.append(f'[@ {path} @] {cmd}\n')
            plot.append(f'[@ {path} @] ./plot.sh\n')
            names.append(path)

        assert len(names) == len(set(names)), "task paths not unique"
        run_path = os.path.join(target_dir, run_tasks_path)
        plot_path = os.path.join(target_dir, 'tasks.plot.txt')
        adaled.save(''.join(run), path=run_path)
        adaled.save(''.join(plot), path=plot_path)
        self.greasy_file_path = run_path
        self.task_names = names

    def format_argv(self, key: str, value: Any):
        return adaled.DataclassMixin.format_argv(key, value)

    def prepare_folders(self, target_dir: str):
        for name in self.task_names:
            os.makedirs(os.path.join(target_dir, name), exist_ok=True)
        print(f"Created folders for {len(self.task_names)} task(s).")

    def prepare_shortcuts(self, target_dir: str):
        path = self.HYPERPARAM_SCRIPT_PATH
        if not path:
            import inspect
            path = inspect.getfile(self.__class__)

        greasy_file = os.path.relpath(self.greasy_file_path, target_dir)
        adaled.save_executable_script(
                os.path.join(target_dir, 'launch_locally.sh'),
                LAUNCH_LOCALLY_SH.format(greasy_file=greasy_file))
        adaled.save_executable_script(
                os.path.join(target_dir, 'plot_all.sh'),
                PLOT_ALL_SH.format(path_prefix=self.PATH_PREFIX))
        adaled.save_executable_script(
                os.path.join(target_dir, 'postprocess.sh'),
                POSTPROCESS_SH.format(path=path))
        adaled.save_executable_script(
                os.path.join(target_dir, 'exec_nth.sh'),
                EXEC_NTH_SH.format(greasy_file=greasy_file))
        adaled.save(' '.join(sys.argv) + '\n',
                    path=os.path.join(target_dir, 'argv.txt'))

    def random_combinations(
            self,
            space: Dict[str, Any],
            N: int,
            random_state: np.random.RandomState,
            repeat: bool = False) -> List[Any]:
        """Generate `N` random combinations in the given parameter space.

        Arguments:
            space: (dict) dictionary of parameter names and their potential
                   values (*)
            N: (int) number of combinations to generate
            repeat: (bool) if True, the same combination is repeated N times,
                    useful for benchmarking (defaults to False)

        (*) The value can be one of the following:
            - scipy probability distribution, e.g. scipy.stats.loguniform(...)
            - list of values, uniformly sampled
            - string, a reference to another key
        """
        combinations = []
        none_count = 0
        while len(combinations) < N:
            comb = {}
            for key, x in space.items():
                if hasattr(x, 'rvs'):  # scipy.stats.randint and similar
                    x = x.rvs(random_state=random_state)
                elif isinstance(x, list):
                    x = x[random_state.choice(len(x))]
                elif isinstance(x, str):  # link to another item
                    x = comb[x]
                else:
                    raise NotImplementedError(x)
                comb[key] = x
            comb = self.generate_sample(
                    kwargs=comb, random_state=random_state,
                    sample_idx=len(combinations), num_samples=N)
            if comb is None:
                none_count += 1
                if none_count == 10000:
                    raise RuntimeError(
                            "generate_sample returned None 10000 times "
                            "in a row, did you forget to return kwargs?")
                continue
            else:
                none_count = 0
            if repeat:
                return [comb] * N
            combinations.append(comb)
        return combinations

    @staticmethod
    def sort_combinations(combinations: List[Any], sort_keys: str) -> List[Any]:
        # sorted() is stable, sort one key at a time, in reverse order.
        for key in reversed(sort_keys.split(',')):
            if not key:
                continue
            if key[0] == '-':
                rev = True
            elif key[0] == '+':
                rev = False
            else:
                raise ValueError("each sort key must start with either "
                                 "+ or -, got {sort_keys!r}")
            key = key[1:]
            combinations = sorted(combinations, key=lambda x: x[key], reverse=rev)
        return combinations

    def postprocess(self, args):
        if args.extra_argv:
            raise ValueError("unexpected extra arguments")
        from adaled.tools.launch_greasy import parse_greasy
        tasks = parse_greasy(os.path.join(args.target_dir, args.run_tasks_path))
        all_keys = {}  # Use dict instead of set to preserve the order somewhat.
        results = []
        for i, task in enumerate(tasks):
            dir = os.path.join(args.target_dir, task.path)
            print(f"Loading {i+1}/{len(tasks)}: {dir}", file=sys.stderr)

            # <dir>/output is legacy, 2022-01-31.
            exc = None
            for dir_ in [dir, os.path.join(dir, 'output')]:
                try:
                    diagnostics = adaled.load_diagnostics(dir_)
                    break
                except Exception as e:
                    exc = e
                    print(f"Error reading {dir}:\n    {repr(exc)}", file=sys.stderr)
                    if not isinstance(e, FileNotFoundError):
                        break
            if exc is not None:
                print(f"Skipping {task.path}.", file=sys.stderr)
                continue

            if args.tmp_diag_rank is not None:
                diagnostics = [diagnostics[args.tmp_diag_rank]]
            elif len(diagnostics) > 1:
                raise NotImplementedError(
                        "postprocessing multi-rank diagnostics not "
                        "implemented yet, specify --tmp-diag-rank")
            argv = shlex.split(task.cmd)
            hyperparams = {}
            for arg in argv:
                if not arg.startswith('--cwd=') and '=' in arg:
                    key, value = arg.split('=', maxsplit=1)
                    hyperparams[key] = value

            post = self.postprocess_diagnostics(
                    dir_, hyperparams=hyperparams,
                    diagnostics=diagnostics, args=args)
            # Use underscore to appear as the first column in hiplot.
            post['_path'] = os.path.normpath(dir)

            results.append(post)
            all_keys.update(results[-1])

        # If different runs don't have the same keys (e.g. probabilistic vs
        # deterministic RNNs), ensure they do have, by filling missing fields
        # with nans.
        results = [{key: result.get(key, np.nan) for key in all_keys.keys()}
                   for result in results]

        path = os.path.join(args.target_dir, 'results.csv')
        with open(path, 'w') as f:
            all_keys = [str(k) for k in all_keys]
            all_keys = [k[:-5] if k.endswith(':json') else k for k in all_keys]
            f.write(','.join(all_keys))
            f.write('\n')
            for value in results:
                def clean(x):
                    if hasattr(x, 'ndim') and x.ndim > 0:
                        key = next(k for k, v in value.items() if v is x)
                        raise TypeError(f"unexpected type {x} for the key {key}")
                    if isinstance(x, (list, tuple)):
                        return '-'.join(map(clean, x))
                    else:
                        x = str(x)
                    return x.replace(',', '-')
                f.write(','.join(map(clean, value.values())))
                f.write('\n')

        sys.stderr.flush()
        print(f"\n\nResults stored to {path}")
        print("To visualize them, run the hiplot webserver:")
        print("    pip install hiplot")
        print("    hiplot")
        print("If the results are stored remotely, enable port forwarding, e.g.:")
        print("    ssh -L 5005:localhost:5005 <remote>")
        print("Open in your browser:")
        print("    http://localhost:5005")
        print(f"and type {path} in the textbox on top.")

    def postprocess_diagnostics(
            self,
            output_dir: str,
            hyperparams: dict,
            diagnostics: Sequence[adaled.TensorCollection],
            args,
            merge_losses: bool = True) -> dict:
        """Return a dictionary of values (single numbers or strings) to plot as
        in e.g. hiplot."""
        post = adaled.postprocess_diagnostics(
                diagnostics, avg_window=args.post_avg_window,
                max_timesteps=args.post_max_timesteps)

        if merge_losses:
            self.post_merge_losses(post)
        # Detect RNN number of layers and hidden size, compute total hiden size.
        rnn_num_layers = None
        rnn_hidden_size = None
        for key, value in hyperparams.items():
            if 'rnn_hidden_size' in key.split('.'):
                rnn_hidden_size = int(value)
            elif 'rnn_num_layers' in key.split('.'):
                rnn_num_layers = int(value)
        if rnn_num_layers is not None and rnn_hidden_size is not None:
            post['total_rnn_hidden_size'] = rnn_num_layers * rnn_hidden_size

        post.update(hyperparams)
        return post

    def post_merge_losses(self, post: dict):
        """Normalize the MSE part of the error, such that deterministic and
        probabilistic show up the same way."""
        keys = [
            ('losses.macro_train.original.mse', 'losses.macro_train'),
            ('losses.macro_valid.mse', 'losses.macro_valid'),
        ]
        for long, short in keys:
            if long in post:
                assert short not in post
                post[short] = post.pop(long)


class SlurmHyperParameterStudy(HyperParameterStudy):
    SLURM_TEMPLATE = None
    DEFAULT_TEMPLATE_ARGS = {}

    def create_parser(
            self, parser: Optional[argparse.ArgumentParser] = None):
        parser = super().create_parser(parser)
        def_ = self.default_kwargs.get
        add = parser.add_argument
        # TODO: add_argument_group
        add('-N', '--nodes', type=int, default=def_('--nodes', 1),
            help="number of nodes")
        add('--time', type=_time_format, default='24:00:00',
            help="time limit in HH:MM:SS")
        add('--ntasks-per-node', type=int, default=def_('--ntasks-per-node', 1),
            help="number of tasks (ranks) per node")
        add('--cpus-per-task', type=int, default=def_('--cpus-per-task', 1),
            help="number of cores per task, skipped if not specified")
        add('--repeat-params', action='store_true', default=False,
            help="use same params in all runs, useful for benchmarking "
                 "multiple tasks per node")
        # Note: we could add a parameter for the slurm template file.
        return parser

    def main_prepare(self, args):
        if not self.SLURM_TEMPLATE:
            raise ValueError("self.SLURM_TEMPLATE empty or not specified")
        super().main_prepare(args)

        greasy_file = os.path.relpath(self.greasy_file_path, args.target_dir)
        name = os.path.normpath(args.target_dir)
        name = os.path.basename(os.path.abspath('.')) \
                if name == '.' else name.replace(os.sep, '_')
        kwargs = self.DEFAULT_TEMPLATE_ARGS.copy()
        kwargs.update(args.__dict__)
        if 'ntasks' not in kwargs:
            kwargs['ntasks'] = int(kwargs['nodes']) * int(kwargs['ntasks_per_node'])
        content = self.SLURM_TEMPLATE.format(
                venv=bash_gen_source_venv(), job_name=name,
                greasy_file=greasy_file,
                combinations_minus_1=args.combinations - 1, **kwargs)
        adaled.save_executable_script(
                os.path.join(args.target_dir, 'job_slurm.sh'),
                content)
        adaled.save_executable_script(
                os.path.join(args.target_dir, 'launch_slurm.sh'),
                LAUNCH_SLURM_SH)
