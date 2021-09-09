from base import TestCase
from adaled.backends import TensorCollection

try:
    import h5py
    from adaled.utils.io_hdf5 import load_hdf5, update_hdf5_group, save_hdf5
except ModuleNotFoundError:
    h5py = None

import numpy as np

import os
import tempfile
import unittest

@unittest.skipIf(h5py is None, "h5py not available")
class TestIO_HDF5(TestCase):
    def test_save_hdf5_and_load_hdf5(self):
        x = np.arange(5)
        y = 123.0 * np.arange(12).reshape(3, 4)
        z = 5
        c = TensorCollection({'a': {'x': x, 'b': {'y': y, 'z': z}}})

        with tempfile.TemporaryDirectory() as dir_:
            path = os.path.join(dir_, 'file.h5')
            save_hdf5(c, path)
            c2 = load_hdf5(path)
            self.assertCollectionEqual(c2, c)

            c3 = load_hdf5(path, lambda obj: 'x' in obj.name)
            self.assertCollectionEqual(c3, {'a': {'x': x}})

            # x[1:], skip y, take z as is
            c4 = load_hdf5(path, lambda obj: (obj[1:] if 'x' in obj.name else None) \
                                             if obj.ndim > 0 else obj)
            self.assertCollectionEqual(c4, {'a': {'x': x[1:], 'b': {'z': z}}})

    def test_update_hdf5(self):
        x1 = np.arange(5)
        y1 = 123.0 * np.arange(12).reshape(3, 4)
        z = 5

        x2 = 10 * x1
        y2 = 0 + y1
        y2[0::2] *= 5
        y2[1::2] *= 10
        c1 = TensorCollection({'a': {'x': x1, 'b': {'y': y1, 'z': z}}})
        c2 = TensorCollection({'a': {'x': x2, 'b': {'y': y2, 'z': z}}})

        with tempfile.TemporaryDirectory() as dir_:
            path = os.path.join(dir_, 'file.h5')
            save_hdf5(c1, path)

            with h5py.File(path, 'r+') as f:
                update_hdf5_group(f, {'a': {'x': x2}})
                update_hdf5_group(f['a']['b'], {'y': y2[0::2]}, slice(0, None, 2))

                # Should work for datasets as well.
                update_hdf5_group(f['a']['b']['y'], y2[1::2], slice(1, None, 2))

                with self.assertRaises(KeyError):
                    update_hdf5_group(f, {'i_do_not_exist': x2})

                with self.assertRaises(TypeError):
                    # /a is a group, not a dataset.
                    update_hdf5_group(f, {'a': x2})

                with self.assertRaises(TypeError):
                    # /a/b/y is a dataset, not a group.
                    update_hdf5_group(f['a']['b'], {'y': {'z': x2}})

            c3 = load_hdf5(path)
            self.assertCollectionEqual(c3, c2)
