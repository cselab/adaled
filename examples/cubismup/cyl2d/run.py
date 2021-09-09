#!/usr/bin/env python3

# Import immediately because cubismup2d is compiled with a newer version of
# OpenMP than the one used by torch.
import cubismup2d as cup2d

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

    # The barrier should hopefully prevent slurm from killing the server
    # process after the client has finished, in case srun ignored the --wait=0
    # argument. (The Ibarrier brings its own problems obviously...)
    barrier_world = world.Dup()
    try:
        main.main(argv, Config, Setup)
    except:
        print("Error. Launching ibarrier.", flush=True)
        barrier_world.Ibarrier()
        raise
    else:
        print("Final Ibarrier start.", flush=True)
        barrier_world.Ibarrier().Wait()
        print("Final Ibarrier end.", flush=True)


if __name__ == '__main__':
    main()
