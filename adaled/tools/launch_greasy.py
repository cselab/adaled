#!/usr/bin/env python3

"""Tiny greasy library.

Used for parsing simple greasy task files and for running tasks locally using
multiprocessing + subprocess.
"""

from colorama import Fore, Style

from typing import Iterable, List, Optional
import argparse
import dataclasses
import os
import re
import subprocess


@dataclasses.dataclass
class Task:
    path: str
    cmd: str


def parse_greasy(tasks_file: str) -> List[Task]:
    """Parse a greasy file. Return a list of Tasks."""
    regex = re.compile(r'\[@ ([^@]*) @\](.*)')

    out = []
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = regex.search(line)
            if not match:
                raise ValueError(f"Unrecognized format: {line}")
            out.append(Task(path=match.group(1), cmd=match.group(2).strip()))
    return out


def run_task(task: Task, prefix: str = ""):
    print(f"{Fore.GREEN}{prefix}Running {task}{Style.RESET_ALL}.")
    os.makedirs(task.path, exist_ok=True)
    try:
        subprocess.check_call(task.cmd, cwd=task.path, shell=True)
    except Exception as e:
        print(f"{Fore.RED}{prefix}Failed {task}{Style.RESET_ALL}")
        raise Exception(task) from e
    if prefix:
        print(f"{prefix}Done!")


def launch_local_greasy(tasks: Iterable[Task], num_jobs: int):
    """Launch greasy tasks locally.

    Arguments:
        tasks: list of Task instances
        num_jobs: number of jobs to run in parallel (with multiprocessing)
    """
    from multiprocessing import Pool
    import time

    t0 = time.time()
    with Pool(num_jobs) as p:
        prefix = [f"{i}/{len(tasks)}: " for i in range(len(tasks))]
        p.starmap(run_task, zip(tasks, prefix), chunksize=1)
    t1 = time.time()
    print(f"Total execution time: {t1 - t0:.3f}s")


def add_arguments(parser: argparse.ArgumentParser):
    add = parser.add_argument
    add('tasks_file', type=str, help="greasy tasks file")
    add('-j', '--jobs', type=int, default=1, help="number of parallel jobs")
    add('--task-id', type=int, help="(optional) run only one specified task")


def do_main(args):
    tasks = parse_greasy(args.tasks_file)
    if args.task_id is None:
        launch_local_greasy(tasks, num_jobs=args.jobs)
    else:
        if args.jobs != 1:
            raise ValueError("--jobs option is incompatible with --task-id")
        run_task(tasks[args.task_id])


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)
    do_main(args)


if __name__ == '__main__':
    main()
