from mpi_base import TestCase, MPI, world
from adaled.backends import TensorCollection
from adaled.utils.data.datasets import CappedTrajectoryDataset, load_dataset
from adaled.utils.data.replacement import RandomReplacementPolicy
from adaled.utils.data.hdf5 import HDF5DynamicTrajectoryDataset

try:
    import h5py
except:
    h5py = None

import numpy as np

from typing import Optional
import os
import functools
import tempfile
import unittest

def TR(trajectory):
    """Make a test collection."""
    return TensorCollection(x=np.asarray(trajectory))


def TRA(n):
    return TensorCollection(x=np.arange(n))


@unittest.skipUnless(h5py, "h5py not available")
class MPITestHDF5DynamicsTrajectoryDataset(TestCase):
    def setUp(self):
        self.dataset = None
        self.comm = None
        self.tmpdir: Optional[tempfile.TemporaryDirectory] = None

    def tearDown(self):
        # Even skipped tests call tearDown. At least those skipped with .skipTest().
        if self.comm:
            self.comm.Barrier()
        if self.dataset:
            # Has to be manually closed to avoid some internal errors.
            self.dataset.f.close()
        if self.tmpdir:
            self.tmpdir.cleanup()

    def make_dataset(self, tmpdir: str, comm):
        return HDF5DynamicTrajectoryDataset(tmpdir, 'dataset.h5', comm=comm)

    def prepare_or_skip(self, size):
        """Prepare the temporary directory and the communicator or skip the test."""
        if self.comm is None:
            self.comm = super().prepare_or_skip(size)
        else:
            assert self.comm.size == size
        comm = self.comm

        if comm.rank == 0:
            self.tmpdir = tempfile.TemporaryDirectory()
            tmpdir = comm.bcast(self.tmpdir.name)
        else:
            tmpdir = comm.bcast(None)
        os.makedirs(tmpdir, exist_ok=True)
        self.dataset = self.make_dataset(tmpdir, comm)
        return self.dataset, tmpdir, comm

    def test_one_round_of_adding(self):
        dataset, tmpdir, comm = self.prepare_or_skip(3)
        self._one_round_of_adding(dataset, comm)

        dataset.f.flush()
        comm.Barrier()
        if comm.rank == 0:
            self.assertHDF5File(dataset.path, {
                'traj-0000000': {'x': [10, 20]},
                'traj-0000001': {'x': [30]},
                'traj-0000002': {'x': [40, 50]},
                'traj-0000003': {'x': [60, 70, 80, 90]},
            })

    def _one_round_of_adding(self, dataset, comm):
        if comm.rank == 0:
            to_add = [TR([10, 20]), TR([30]), TR([40, 50])]
        elif comm.rank == 1:
            to_add = []
        else:
            to_add = [TR([60, 70, 80, 90])]

        dataset.add_trajectories(to_add)
        self.assertEqual(len(dataset.as_trajectories()), len(to_add))
        for current, expected in zip(dataset.as_trajectories(), to_add):
            self.assertCollectionEqual(current, expected)

    def _multiple_rounds_of_adding(self, dataset, comm):
        if comm.rank == 0:
            to_add1 = [TRA(10), TRA(20)]
            to_add2 = [TRA(40), TRA(50)]
        elif comm.rank == 1:
            to_add1 = []
            to_add2 = [TRA(60)]
        else:
            to_add1 = [TRA(30)]
            to_add2 = [TRA(70), TRA(80), TRA(90)]

        dataset.add_trajectories(to_add1)
        dataset.add_trajectories(to_add2)
        to_add = to_add1 + to_add2

        self.assertEqual(len(dataset.as_trajectories()), len(to_add))
        for current, expected in zip(dataset.as_trajectories(), to_add):
            self.assertCollectionEqual(current, expected)

    def test_multiple_rounds_of_adding(self):
        """Test invoking add_trajectories multiple times, with 0-dim data."""
        dataset, tmpdir, comm = self.prepare_or_skip(3)
        self._multiple_rounds_of_adding(dataset, comm)

        # Test whole dataset.
        if comm.rank == 0:
            expected = {f'traj-{i:07d}': {'x': np.arange(10 * (i + 1))} for i in range(9)}
            self.assertHDF5File(dataset.path, expected)

        self._multiple_rounds_of_adding_test_update(dataset, comm)

    def _multiple_rounds_of_adding_test_update(self, dataset, comm):
        states = dataset.as_states()
        traj = dataset.as_trajectories()

        # Test update.
        if comm.rank == 0:
            dataset.set_trajectories([1], [TRA(100)])
            self.assertEqual(len(states), 10 + 100 + 40 + 50)
            self.assertCollectionEqual(traj[1], TRA(100))
        elif comm.rank == 1:
            dataset.set_trajectories([0], [TRA(110)])
            self.assertEqual(len(states), 110)
            self.assertCollectionEqual(traj[0], TRA(110))
        else:
            dataset.set_trajectories([2], [TRA(120)])
            self.assertEqual(len(states), 30 + 70 + 120 + 90)
            self.assertCollectionEqual(traj[2], TRA(120))

    def test_with_tensor_collection(self):
        # Note that trajectories stored on one rank may be inaccessible from
        # that rank later because of distributed nature of the dataset.
        dataset, tmpdir, comm = self.prepare_or_skip(2)
        xy = np.arange(5).reshape(1, 5)
        dataset.add_trajectories(TensorCollection({'x': {'y': xy}}))

        self.assertCollectionEqual(dataset.as_trajectories()[0], {'x': {'y': xy[0]}})
        self.assertCollectionEqual(dataset.as_trajectories('x')[0], {'y': xy[0]})
        self.assertArrayEqual(dataset.as_trajectories(('x', 'y'))[0], xy[0])

        self.assertCollectionEqual(dataset.as_states()[2], {'x': {'y': xy[0, 2]}})
        self.assertCollectionEqual(dataset.as_states('x')[2], {'y': xy[0, 2]})
        self.assertArrayEqual(dataset.as_states(('x', 'y'))[2], xy[0, 2])

    def test_save_load(self):
        dataset, tmpdir, comm = self.prepare_or_skip(3)

        if comm.rank == 0:
            dataset.add_trajectories([TRA(10), TRA(20)])
            dataset.add_trajectories([TRA(40), TRA(50)])
            self.assertArrayEqual(dataset._local_to_global, [0, 1, 3, 4])
        elif comm.rank == 1:
            dataset.add_trajectories([])
            dataset.add_trajectories([TRA(60)])
            self.assertArrayEqual(dataset._local_to_global, [5])
        else:
            dataset.add_trajectories([TRA(30)])
            dataset.add_trajectories([TRA(70), TRA(80), TRA(90)])
            self.assertArrayEqual(dataset._local_to_global, [2, 6, 7, 8])

        def _test_load(tmpdir: str, num_ranks: int):
            comm2 = comm.Split(comm.rank < num_ranks)
            if comm.rank >= num_ranks:
                return
            dataset2 = load_dataset(tmpdir, comm=comm2)
            all_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            expected_sizes = all_sizes[comm.rank::num_ranks]
            self.assertEqual(len(dataset2.as_trajectories()), len(expected_sizes))
            for i, size in enumerate(expected_sizes):
                self.assertCollectionEqual(dataset2.as_trajectories()[i], TRA(size))

        # Test save and load. The trajectories are rearranged after load.
        tmpdir = os.path.join(tmpdir, "tmpdir")
        dataset.save(tmpdir, verbose=False, hardlink=False)
        _test_load(tmpdir, 1)
        _test_load(tmpdir, 2)
        _test_load(tmpdir, 3)

        tmpdir = os.path.join(tmpdir, "tmpdir")
        dataset.save(tmpdir, verbose=False, hardlink=True)
        _test_load(tmpdir, 3)
        self.assertTrue(os.path.samefile(
                os.path.join(tmpdir, dataset.filename), dataset.path))

    def test_capped_dataset(self):
        for seed in range(10000, 10010):
            self._test_capped_dataset(capacity=2, num_ranks=2, iterations=10, seed=seed)

    def _test_capped_dataset(self, capacity: int, num_ranks: int, iterations: int, seed: int):
        hdf5_dataset, tmpdir, comm = self.prepare_or_skip(num_ranks)

        rng = np.random.default_rng(seed + comm.rank)
        dataset = CappedTrajectoryDataset(
                hdf5_dataset, RandomReplacementPolicy(rng), capacity, comm)
        for i in range(iterations):
            trajectories = [
                TensorCollection(x=rng.uniform(0.0, 1.0, rng.integers(2, 5)))
                for i in range(rng.integers(3))
            ]
            dataset.add_trajectories(trajectories)

        self.tearDown()
