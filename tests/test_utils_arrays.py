from base import TestCase
from adaled.backends import TensorCollection
from adaled.utils.arrays import join_sequences, masked_gaussian_filter1d, rolling_average

import numpy as np

import pickle

class TestJoinSequences(TestCase):
    def test_equal_length_no_gaps(self):
        x = np.arange(12).reshape(3, 4)
        self.assertArrayEqual(join_sequences(x), x.ravel())

        x = np.arange(30).reshape(5, 2, 3)
        self.assertArrayEqual(join_sequences(x), x.reshape(10, 3))

    def test_equal_length_gaps(self):
        x = np.arange(12).reshape(3, 4)
        self.assertArrayEqual(
                join_sequences(x, gap=-1),
                [0, 1, 2, 3, -1, 4, 5, 6, 7, -1, 8, 9, 10, 11])

        x = np.arange(12).reshape(3, 2, 2)
        computed = join_sequences(x, gap=[-1, -2])
        expected = [
            [0, 1],
            [2, 3],
            [-1, -2],
            [4, 5],
            [6, 7],
            [-1, -2],
            [8, 9],
            [10, 11],
        ]
        self.assertArrayEqual(computed, expected)

        x = np.arange(12).reshape(3, 2, 2)
        computed = join_sequences(x, gap=-1)
        expected = [
            [0, 1],
            [2, 3],
            [-1, -1],
            [4, 5],
            [6, 7],
            [-1, -1],
            [8, 9],
            [10, 11],
        ]
        self.assertArrayEqual(computed, expected)

    def test_varying_length_without_gaps(self):
        x = [np.array([1]), np.array([2, 3]), np.array([4, 5, 6])]
        self.assertArrayEqual(join_sequences(x), [1, 2, 3, 4, 5, 6])

    def test_varying_length_with_gap(self):
        x = [np.array([1]), np.array([2, 3]), np.array([4, 5, 6])]
        self.assertArrayEqual(join_sequences(x, gap=-1),
                              [1, -1, 2, 3, -1, 4, 5, 6])

    def test_with_tensor_collection_equal_len_with_gap(self):
        x = TensorCollection(a=np.arange(6).reshape(3, 2),
                             b=np.arange(24).reshape(3, 2, 4))
        computed = join_sequences(x, gap=-1)
        expected_a = np.array([0, 1, -1, 2, 3, -1, 4, 5])
        expected_b = np.array([
            [0, 1, 2, 3], [4, 5, 6, 7], [-1, -1, -1, -1],
            [8, 9, 10, 11], [12, 13, 14, 15], [-1, -1, -1, -1],
            [16, 17, 18, 19], [20, 21, 22, 23],
        ])
        self.assertCollectionEqual(computed, {'a': expected_a, 'b': expected_b})

    def test_with_tensor_collection_varying_length_with_gap(self):
        x0 = TensorCollection(a=np.arange(5), b=np.arange(10).reshape(5, 2))
        x1 = TensorCollection(a=np.arange(3), b=np.arange(6).reshape(3, 2))
        x2 = TensorCollection(a=np.arange(4), b=np.arange(8).reshape(4, 2))
        computed = join_sequences([x0, x1, x2], gap=-1)
        expected_a = np.array([0, 1, 2, 3, 4, -1, 0, 1, 2, -1, 0, 1, 2, 3])
        expected_b = np.array([
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [-1, -1],
            [0, 1], [2, 3], [4, 5], [-1, -1],
            [0, 1], [2, 3], [4, 5], [6, 7],
        ])
        self.assertCollectionEqual(computed, {'a': expected_a, 'b': expected_b})


class TestArrayUtils(TestCase):
    def test_masked_gaussian_filter(self):
        from scipy.ndimage import gaussian_filter1d  # Lazily import scipy.
        a = np.array([10., 11., 15., 20., 21., 22., 23., 25., 10.])
        self.assertArrayAlmostEqual(masked_gaussian_filter1d(a, 2.0),
                                    gaussian_filter1d(a, 2.0))

        out = a.copy()
        masked_gaussian_filter1d(a, 2.0, out=out)
        self.assertArrayAlmostEqual(out, gaussian_filter1d(a, 2.0))

        a = [3., 3., np.nan, 3., np.nan, 3., 3.]
        self.assertArrayAlmostEqual(masked_gaussian_filter1d(a, 2.0),
                                    [3., 3., 3., 3., 3., 3., 3.])
        self.assertArrayAlmostEqual(masked_gaussian_filter1d(a, 2.0, set_nan=True), a)

    def test_rolling_average(self):
        self.assertArrayEqual(
                rolling_average([10., 20., 20., 50.], 2),
                [10., 15., 20., 35.])
        self.assertArrayEqual(
                rolling_average([10., 20., 30., 50., 60., 70.], 4),
                [10., 15., 20., 27.5, 40., 52.5])
        self.assertArrayEqual(
                rolling_average([True, True, True, False, True, True, False], 4),
                [1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.5])

        # Test n less than the size of the array.
        self.assertArrayEqual(
                rolling_average([10, 20., 30.], 10),
                [10., 15., 20.])
