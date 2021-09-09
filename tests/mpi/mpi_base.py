import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

from base import TestCase as _TestCase

from mpi4py import MPI

from typing import Optional
import traceback
import unittest

world: MPI.Intracomm = MPI.COMM_WORLD


class TestCase(_TestCase):
    def _callTestMethod(self, method):  # Override.
        """Detect exceptions thrown by the tests, print them manually and
        abort. Used to avoid deadlocks and missing output on failure."""
        try:
            method()
        except unittest.SkipTest:
            raise
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort()

    def _prepare_or_skip(self, size: int):
        """Prepare the communicator of required size or skip the test.

        The test is skipped if the world communicator is too small or if this
        rank is not within the [0..size-1] range.
        """
        if world.size < size:
            self.skipTest("too few ranks")
        comm = world.Split(world.rank < size)
        if world.rank >= size:
            self.skipTest(f"rank {world.rank} not used, only {size} needed, "
                          f"world.size={world.size}")
        return comm

    prepare_or_skip = _prepare_or_skip

    def prepare_intercomm_or_skip(self, size1: int, size2: int):
        comm = self._prepare_or_skip(size1 + size2)
        is_first = comm.rank < size1
        localcomm: MPI.Intracomm = comm.Split(is_first)
        intercomm: MPI.Intercomm = \
                localcomm.Create_intercomm(0, comm, (size1 if is_first else 0))
        return comm, localcomm, intercomm, is_first

    def assertComm(self,
                   comm,
                   size: Optional[int] = None,
                   rank: Optional[int] = None,
                   inter: Optional[bool] = None,
                   remote_size: Optional[int] = None):
        """Assert if comm size and rank are correct."""
        if size is not None:
            self.assertEqual(comm.size, size)
        if rank is not None:
            self.assertEqual(comm.rank, rank)
        if inter is not None:
            self.assertEqual(comm.is_inter, inter)
        if remote_size:
            self.assertTrue(comm.is_inter)
            self.assertEqual(comm.remote_size, remote_size)

    def assertCommNull(self, comm):
        """Assert that the communicator is equal to MPI.COMM_NULL."""
        self.assertEqual(comm, MPI.COMM_NULL)


class TextTestResult(unittest.TextTestResult):
    def printErrorList(self, flavour, errors):
        flavour = f"{flavour} (rank={world.rank})"
        super().printErrorList(flavour, errors)


class TestRunner(unittest.TextTestRunner):
    resultclass = TextTestResult


if __name__ == '__main__':
    unittest.main(module=None, testRunner=TestRunner)
