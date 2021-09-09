"""
The task system below is motivated by the unittest package.

We could in principle use unittest.TestCase and unittest.TestLoader as a base
for implementing tasks, but that might lead to confusion because unittest case
is not entirely generic. For example, it is not possible to customize which
class is the TestCase class.

Another difference is that we want to parallelize plotting.

The discovery code now looks for functions named `tasks_*`, which are invoked
only to get the list of `Task` objects. These tasks are then executed in
parallel.
"""

from fnmatch import fnmatchcase
from typing import Callable, Iterable, List, Optional, Tuple
import argparse
import inspect
import os

from adaled.plotting.plots import Plot

class SkipTaskException(Exception):
    pass


class Task:
    __slots__ = ('fn', 'args', 'kwargs', 'no_parallel')

    def __init__(self, fn, *args, _task_no_parallel: bool = False, **kwargs):
        """
        Arguments:
            fn: (callable) function to invoke
            args: (tuple) position arguments to pass to `fn`
            kwargs: (dict) keyword arguments to pass to `fn`
            _task_no_parallel: (bool) If set, run outside of the
                multiprocessing pool. Useful when the task itself uses
                multiprocessing internally or parallelizes the job in another
                fashion. (default: `False`)
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.no_parallel = _task_no_parallel

    def __repr__(self):
        return f"Task({self.fn}, ...)"

    def __call__(self):
        try:
            ret = self.fn(*self.args, **self.kwargs)
            if hasattr(ret, '_finalize_task'):
                ret._finalize_task()
        except:
            print("Failed task info:")
            print(f"    function={self.fn}")
            def sane_output(value):
                s = str(value)
                return value.__class__ if len(s) > 200 else s

            for i, arg in enumerate(self.args):
                print(f"    arg[{i}]={sane_output(arg)}")
            for key, value in self.kwargs.items():
                print(f"    kwarg[{key}]={sane_output(value)}")
            raise

    def _verbose_call(self):
        print(self)
        return self()


def _preprocess_task_name_pattern(exclude_prefix: str):
    def convert(pattern: str):
        # Contrary to unittest, we always put * at the beginning and end.
        if pattern.startswith(exclude_prefix):
            return f'{exclude_prefix}*{pattern[len(exclude_prefix):]}*'
        else:
            return f'*{pattern}*'

    return convert


class TaskCase:  # Analoguous of unittest.TestCase.
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        pass  # Default is no-op.

    @classmethod
    def main(cls, argv: Optional[List[str]] = None):
        """Discover tasks of the current class and run them."""
        discover_and_run([cls], argv)

    def task_subcases(self):
        """For nested tasks cases, return the list of cases to take tasks from."""
        return []

    def set_up(self):
        """Invoked before each tasks_* function that was not filtered out."""
        pass  # No-op by default.

    def skip(self, msg: Optional[str] = None):
        """Skip the current tasks_* function with the given explanation."""
        raise SkipTaskException(msg)


class TaskSuite:
    # Would use an '!' but it has a special meaning in bash.
    EXCLUDE_PREFIX = '@'

    def add_arguments(self, parser: argparse.ArgumentParser):
        add = parser.add_argument_group('tasks').add_argument
        add('-j', '--jobs', type=int, help="number of parallel jobs")
        add('-v', '--verbose', action='store_true', default=False)
        add('-k', dest='task_name_patterns', action='append',
            type=_preprocess_task_name_pattern(self.EXCLUDE_PREFIX),
            default=[], metavar="PATTERN",
            help=f"Run only tasks that match given pattern(s). Use a prefix "
                 f"'{self.EXCLUDE_PREFIX}' to skip tests instead.")
        add('-l', '--list', action='store_true', default=False,
            help="print all available plots")
        return parser

    def discover_task_fns_in_case(self, case):
        cls = case.__class__
        module = inspect.getmodule(cls)
        module_name = module.__name__
        if module_name == '__main__':
            path = inspect.getsourcefile(module)
            module_name = os.path.splitext(os.path.basename(path))[0]

        task_fns = []
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith('tasks_'):
                # TODO: Add file name.
                qualname = f'{module_name}.{cls.__name__}.{name}'
                task_fns.append((case, fn, qualname))

        for child_case in case.task_subcases():
            task_fns += self.discover_task_fns_in_case(child_case)

        return task_fns

    def filter_task_fns(self,
                        task_fns: List[Tuple[str, Callable]],
                        task_name_patterns: List[str]):
        for pattern in task_name_patterns:
            exclude = self.EXCLUDE_PREFIX \
                    and pattern.startswith(self.EXCLUDE_PREFIX)
            if exclude:
                pattern = pattern[len(self.EXCLUDE_PREFIX):]
            task_fns = [
                (case, fn, qualname)
                for case, fn, qualname in task_fns
                if bool(fnmatchcase(qualname, pattern)) != exclude
            ]
        return task_fns

    def to_tasks(self, task_fns):
        out = []
        for case, fn, qualname in task_fns:
            try:
                case.set_up()
                out.extend(fn(case))
            except SkipTaskException as e:
                print(f"SKIP {qualname}: {e.args[0]}")
        return out

    def run(self, task_fns, args):
        task_fns = self.filter_task_fns(task_fns, args.task_name_patterns)
        if args.list:
            print("Available tasks:")
            for case, fn, qualname in task_fns:
                print(qualname)
        else:
            self.run_tasks(task_fns, args)

    def run_tasks(self, task_fns, args):
        tasks = self.to_tasks(task_fns)
        run_tasks(tasks, jobs=args.jobs, verbose=args.verbose)


def discover_and_run(classes: List[type], argv: Optional[List[str]] = None):
    """Discover tasks in given TaskCase classes and run them."""
    parser = argparse.ArgumentParser()

    suite = TaskSuite()
    suite.add_arguments(parser)
    for cls in classes:
        cls.add_arguments(parser)

    args = parser.parse_args(argv)

    tasks = []
    for cls in classes:
        plotter = cls(args)
        tasks.extend(suite.discover_task_fns_in_case(plotter))
    suite.run(tasks, args)


def run_tasks(tasks: Iterable[Task], jobs: Optional[int] = None, verbose: bool = False):
    """Run given tasks in parallel.

    If the number of jobs is not specified it is set automatically to
    `os.cpu_count()`."""
    tasks: List[Task] = list(tasks)
    if not int(os.environ.get('ADALED_NO_TASKS_MPI') or '0'):
        try:
            from mpi4py import MPI
            world = MPI.COMM_WORLD
        except ImportError:
            world = None
    else:
        world = None

    if world and world.size > 1:
        tasks = tasks[world.rank::world.size]
        print(f"Running {len(tasks)} task(s) on rank #{world.rank}...", flush=True)
    else:
        print(f"Running {len(tasks)} task(s)...", flush=True)

    parallel = [task for task in tasks if not task.no_parallel]
    non_parallel = [task for task in tasks if task.no_parallel]

    _run(parallel, jobs, verbose)
    _run(non_parallel, 1, verbose)
    print("Done!")


def _run(tasks: List[Task], jobs: int, verbose: bool):
    if jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            if verbose:
                print(task)
            out = task()
            if isinstance(out, Plot):
                out.finalize()
    else:
        from multiprocessing import Pool
        with Pool(processes=jobs) as p:
            if verbose:
                p.map(Task._verbose_call, tasks, chunksize=1)
            else:
                p.map(Task.__call__, tasks, chunksize=1)
