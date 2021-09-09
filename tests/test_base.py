"""Test the base.py itself."""
from base import TestCase

import numpy as np

import os
import tempfile

class TestHDF5Asserts(TestCase):
    def test_hdf5_asserts(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not available")
        from adaled.utils.io_hdf5 import H5PyFile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tmp.h5')
            with H5PyFile(path, 'w') as f:
                d1 = f.create_dataset('dataset1', data=np.arange(5))
                g1 = f.create_group('group1')
                d2 = g1.create_dataset('dataset2', data=np.arange(3))

            expected = {
                'dataset1': [0, 1, 2, 3, 4],
                'group1': {'dataset2': [0, 1, 2]},
            }
            self.assertHDF5File(path, expected)
