#!/usr/bin/env python3

# Import immediately because libcubismup3d is compiled with a newer version of
# OpenMP than the one used by torch.
import libcubismup3d as cup3d

from mpi4py import MPI

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main(argv=None, world: MPI.Intracomm = MPI.COMM_WORLD):
    """Parse command-line arguments and invoke the simulation main function.

    The purpose of this function and file is to avoid having any classes in the
    `__main__` module, which breaks unpickling.
    """
    from ..common.setup import ServerClientMainEx
    from .setup import Config, Setup
    main = ServerClientMainEx(world)
    main.main(argv, Config, Setup)


if __name__ == '__main__':
    main()
