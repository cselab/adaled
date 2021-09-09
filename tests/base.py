import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # Repo root.

import numpy as np

REGRESSION_DATA_DIR = os.path.join(os.path.dirname(__file__), 'regression_data')

from adaled import TensorCollection
from adaled.backends import cforeach
from adaled.utils.data.datasets import TrajectoryDataset
import adaled

from typing import Dict, Tuple, Union

class TestCase(unittest.TestCase):
    def assertRegressionCSV(self, name, mat, *, save=False, header=None, fmt='%.9g'):
        """Check if the 2D array `mat` matches the values stored in the
        `<REGRESSION_DATA_DIR>/<name>`.

        If `save` is set to `True`, the CSV file is stored first before loading it.
        (The loading and comparing is done nevertheless to verify that the
        stored precision is sufficient for the test to pass.)

        If a test fails, the matrix `mat` is saved to
        `_incorrect_test_results.csv'.
        """

        path = os.path.join(REGRESSION_DATA_DIR, name)
        if save:
            np.savetxt(path, mat, delimiter=',', fmt=fmt, header=header, comments='')
            print(f"WARNING: overwritting the expected result `{path}`. "
                  f"Don't forget to reset the parameter `save` to `False`!")

        expected = np.loadtxt(path, skiprows=(1 if header else 0), delimiter=',')
        try:
            np.testing.assert_almost_equal(mat, expected)
        except AssertionError as e:
            out = '_incorrect_test_result.csv'
            np.savetxt(out, mat, delimiter=',', fmt='%.9g')
            print(f"Expected values: {path}")
            print(f"Computed values: {out}")
            raise e

    def assertArrayEqual(self, a, b, *args,
                         check_backend: bool = False,
                         check_shape: bool = False,
                         approx: bool = False,
                         **kwargs):
        if check_backend:
            self.assertEqual(adaled.get_backend(a), adaled.get_backend(b))
        a = adaled.to_numpy_nonstrict(a)
        b = adaled.to_numpy_nonstrict(b)
        if check_shape:
            self.assertEqual(a.shape, b.shape)
        if approx:
            np.testing.assert_almost_equal(a, b, *args, **kwargs)
        else:
            np.testing.assert_array_equal(a, b, *args, **kwargs)

    def assertArrayEqualStrict(self, a, b, *args, **kwargs):
        self.assertIs(a.__class__, b.__class__)
        self.assertEqual(a.dtype, b.dtype)
        if hasattr(a, 'device'):
            self.assertEqual(a.device, b.device)
            a = a.cpu()
            b = b.cpu()
        np.testing.assert_array_equal(a, b, *args, **kwargs)

    def assertArrayAlmostEqual(self, a, b, *args, approx: bool = True, **kwargs):
        """Alias of `self.assertArrayEqual(..., approx=True, ...)`."""
        self.assertArrayEqual(a, b, *args, approx=approx, **kwargs)

    def assertCollectionEqual(self, a, b):
        a = TensorCollection(a)
        b = TensorCollection(b, default_numpy=True)
        self.assertEqual(a.hierarchy(), b.hierarchy())

        def cmp(key, value_a):
            value_b = b[key]
            try:
                len(value_a)
                has_length = True
            except TypeError:
                has_length = False

            if has_length:
                self.assertArrayEqual(value_a, value_b, "/".join(key))
            else:
                self.assertEqual(value_a, value_b, key)

        a.named_foreach(cmp)

    def assertDatasetEqual(self, dataset1: TrajectoryDataset, dataset2: TrajectoryDataset):
        def _test(data1, data2):
            self.assertEqual(len(data1), len(data2))
            for item1, item2 in zip(data1, data2):
                is_collection1 = isinstance(item1, (dict, TensorCollection))
                is_collection2 = isinstance(item2, (dict, TensorCollection))
                self.assertEqual(is_collection1, is_collection2)
                if is_collection1:
                    self.assertCollectionEqual(item1, item2)
                else:
                    self.assertArrayEqual(item1, item2)

        def _print(dataset, name):
            print(f"Dataset {name} information:")
            print("Number of trajectories:", len(dataset.as_trajectories()))
            print("Number of states:", len(dataset.as_states()))
            for traj in dataset.as_trajectories():
                print(traj.shape)

        # Compare both trajectories and states, to test datasets themselves.
        try:
            _test(dataset1.as_trajectories(), dataset2.as_trajectories())
            _test(dataset1.as_states(), dataset2.as_states())
        except:
            _print(dataset1, "1")
            _print(dataset2, "2")
            raise

    def assertEqualModuleStructures(self, a_list, b_list):
        """Assert that two lists of modules have the same structure.

        The module parameters are NOT compared.
        """
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            # Heuristics to compare the modules.
            # Not really guaranteed to always work.
            if str(a) != str(b):
                print(f"Mismatch between modules at layer #{i}.")
                print("Module A:")
                print(a_list)
                print("Module B:")
                if isinstance(b_list, list):
                    import torch
                    b_list = torch.nn.ModuleList(b_list)
                print(b_list)
                self.assertEqual(str(a), str(b))
        self.assertEqual(len(a_list), len(b_list))

    def assertHDF5File(self, path: str, data: dict):
        """Assert that the HDF5 file at the given path has structure and data
        matching the given `data` dictionary."""
        from adaled.utils.io_hdf5 import H5PyFile
        with H5PyFile(path, 'r') as f:
            self.assertHDF5Group(f, data)

    def assertHDF5Group(self, group: 'h5py.Group', data: dict):
        """Assert that the HDF5 group has the structure and data matching the
        given `data` dictionary."""
        self.assertEqual(sorted(group.keys()), sorted(data.keys()))
        for key, expected in data.items():
            group_data = group[key]
            if isinstance(expected, (dict, TensorCollection)):
                self.assertHDF5Group(group_data, expected)
            elif isinstance(expected, (list, np.ndarray)):
                self.assertArrayEqual(expected, group_data[()])
            else:
                raise TypeError("unexpected type: " + expected.__class__.__name__)

    def assertShape(self, array, shape: Union[Tuple[int, ...], Dict], *args, **kwargs):
        self.assertEqual(array.shape, shape, *args, **kwargs)


class TestCaseWithTmpdir(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        super().tearDown()
        if getattr(self, '_tmpdir', None):
            self._tmpdir.cleanup()
