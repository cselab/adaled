from base import TestCase
import adaled.postprocessing.statistics as stats

import numpy as np

import os

class TestPostprocessingStatistics(TestCase):
    def test_compute_1d_channel_stats(self):
        num_batches = 10
        batch_size = 16
        num_channels = 4
        state_size = 20
        num_bins = 5
        ranges = np.array([[-1.0, -3.0, 5.0, 10.0], [+1.0, +3.0, 10.0, 100.0]]).T
        data = np.random.uniform(
                ranges[:, 0, np.newaxis],
                ranges[:, 1, np.newaxis],
                (num_batches * batch_size, num_channels, state_size))
        for i, (vmin, vmax) in enumerate(ranges):
            self.assertTrue((vmin - 1e-9 <= data[:, i]).all())
            self.assertTrue((data[:, i] <= vmax + 1e-9).all())

        batches = data.reshape(num_batches, batch_size, num_channels, state_size)
        computed = stats.compute_1d_channel_stats(
                batches, num_bins, ranges, verbose=False)

        # Test means.
        expected = data.mean(axis=0)
        self.assertArrayAlmostEqual(computed.means, expected, check_shape=True)

        # Test histograms.
        expected = np.zeros((num_channels, state_size, num_bins))
        for channel in range(num_channels):
            for i in range(state_size):
                expected[channel, i], bin_edges = np.histogram(
                        data[:, channel, i], num_bins, ranges[channel])
        expected = expected.astype(np.int64)

        self.assertArrayEqualStrict(computed.histograms, expected)
        self.assertArrayEqual(computed.histogram_ranges, ranges)
