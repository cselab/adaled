import argparse
import json
import os

from adaled.led.topology import Topology, topology_split_comm
from adaled.utils.dataclasses_ import dataclass
from typing import Any, List, Optional

__all__ = ['Main', 'ServerClientMain']

def _multirank_simulation_topology(n: int) -> str:
    """Create a string representing the multirank simulation topology, where
    servers and clients alternate.

    For example:
        >>> _multirank_simulation_topology(3)
        'S0,C0->S0+S1+S2,S1,C0,S2,C0'
    """
    s1 = [f'+S{i}' for i in range(1, n)]
    cs1 = [f',S{i},C0' for i in range(1, n)]
    topology = 'S0,C0->S0' + ''.join(s1) + ''.join(cs1)
    return topology


# FIXME: Some better name for this? This class is reponsible only for reading
# cmdline args and setting up the environment, there is no "main" here.
class Main:
    def __init__(self, world: Optional['Intracomm'] = None):
        self.args = None
        self.world = world

    def add_arguments(self, parser: argparse.ArgumentParser):
        add = parser.add_argument
        add('--config-json', type=str, action='append', default=[],
            help="modifications to apply to the default config")
        add('--cwd', type=str, default='output',
            help="change working directory to the given folder")
        add('--seed', type=int, default=12345,
            help="numpy and torch seeds, -1 to not set seeds")
        add('--seed-rank', type=int, default=134917,
            help="adds --seed-rank * <current rank> to --seed (if seed != -1)")

    def parse_and_process(self, argv: Optional[List[str]], config: Any,
                          parser: Optional[argparse.ArgumentParser] = None):
        """Parse command line arguments and process them on the given `config`.

        Modifies and returns `config`.
        """
        if parser is None:
            parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        self.args, unknown = parser.parse_known_args(argv)
        self.process(self.args, config)
        config.apply_argv(unknown)
        return config

    def process(self, args, config: Any):
        for mod in args.config_json:
            config.update(json.loads(mod))

        if args.cwd:
            os.makedirs(args.cwd, exist_ok=True)
            os.chdir(args.cwd)

        if args.seed != -1:
            import numpy as np
            import torch
            # Ensure that each rank has its own seed. In case consistent random
            # numbers across the communicator are required, the user must handle
            # them manually.
            rank = self.world.rank if self.world else 0
            seed = args.seed + rank * args.seed_rank
            np.random.seed(seed)
            torch.manual_seed(seed)


class ServerClientMain(Main):
    def add_arguments(self, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_mutually_exclusive_group()
        add = group.add_argument
        add('--ranks-per-simulation', type=int, default=1)
        add('--topology', type=str,
            help="manual specification of the server-client topology")

    def init_topology(self, world):
        args = self.args
        if args.topology is not None:
            topology = args.topology
        else:
            topology = _multirank_simulation_topology(args.ranks_per_simulation)
        return topology_split_comm(world, topology)


# Deprecated.
def parse_and_process_argv(argv: Optional[List[str]]=None, *, config: Any):
    import warnings
    warnings.warn("parse_and_process_argv is deprecated")

    main = Main()
    main.parse_and_process(argv, config)
