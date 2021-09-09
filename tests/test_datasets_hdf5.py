from base import TestCaseWithTmpdir
from adaled.backends import TensorCollection
from adaled.utils.data.datasets import load_dataset
from tests.test_datasets import UniformTrajectoryLengthDatasetTests

try:
    import h5py
except ImportError:
    h5py = None
else:
    from adaled.utils.data.hdf5 import HDF5DynamicTrajectoryDataset

import numpy as np

import os
import unittest

def TR(array):
    return TensorCollection(x=np.asarray(array))


@unittest.skipIf(h5py is None, "h5py not available")
class TestHDF5Dataset(UniformTrajectoryLengthDatasetTests, TestCaseWithTmpdir):
    """Single file for the whole dataset, but different groups per trajectory."""
    SUPPORTS_PURE_NDARRAY = False

    def setUp(self):
        super().setUp()
        self.dataset = HDF5DynamicTrajectoryDataset(
                self.tmpdir, 'dataset.h5', traj_fmt='/traj-{:03d}')

    def test_add(self):
        dataset = self.dataset
        # One state is a 2x5 matrix.
        dataset.add_trajectories([
            TR(np.arange(60).reshape(6, 2, 5)),
            TR(np.arange(70).reshape(7, 2, 5)),
        ])
        dataset.add_trajectories([
            TR(np.arange(80).reshape(8, 2, 5)),
            TR(np.arange(90).reshape(9, 2, 5)),
        ])

        self.assertArrayEqual(dataset._local_lengths, [6, 7, 8, 9])
        self.assertArrayEqual(dataset._local_to_global, [0, 1, 2, 3])
        expected = {
            f'traj-{t:03d}': TR(np.arange(10 * (6 + t)).reshape(6 + t, 2, 5))
            for t in range(4)
        }
        self.assertHDF5File(os.path.join(self.tmpdir, 'dataset.h5'), expected)

    def test_load(self):
        dataset = self.dataset
        # One state is a 2x5 matrix.
        dataset.add_trajectories([
            TR(np.arange(60).reshape(6, 2, 5)),
            TR(np.arange(70).reshape(7, 2, 5)),
        ])
        dataset.add_trajectories([
            TR(np.arange(80).reshape(8, 2, 5)),
            TR(np.arange(90).reshape(9, 2, 5)),
        ])

        dir2 = os.path.join(self.tmpdir, 'copy')
        dataset.save(dir2, verbose=False)

        dataset2 = load_dataset(dir2)
        self.assertDatasetEqual(dataset, dataset2)
