import adaled
import adaled.utils.data.datasets as datasets
import adaled.utils.io_ as io_

import numpy as np
import torch
import tqdm

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

@dataclass
class Channels1DStats:
    # (num_channels, state_size), float
    means: np.ndarray

    # (num_channels, state_size, num_bins), int
    histograms: np.ndarray

    # (num_channels, 2), float
    histogram_ranges: np.ndarray

    # (num_channels, num_bins + 1), float
    histogram_bins: np.ndarray


@dataclass
class PostprocessedData1D:
    """Various postprocessed data for generic multi-channel 1D systems."""
    original: Channels1DStats
    reconstructed: Channels1DStats
    num_trajectories: int
    num_samples: int


def batched_2d_densities(
        dataloader: Iterable[Sequence[Tuple[np.ndarray, np.ndarray]]],
        nbins: Union[int, Tuple[int, int]],
        range: np.ndarray):
    """Compute 2D density histograms of multiple batched data.

    Suitable for computing different histograms on large datasets.

    Arguments:
        dataloader: iterable of [(x, y), ...] batches, where each (x, y)
                    "column" denotes one histogram
        nbins: number of bins per dimension
        range: ((xmin, xmax), (ymin, ymax))

    Returns:
        density array of shape (number of histograms, nbins_x, nbins_y)
    """
    counts = None
    for xys in dataloader:
        for i, (x, y) in enumerate(xys):
            x = adaled.to_numpy(x)
            y = adaled.to_numpy(y)
            assert len(x) == len(y), (len(x), len(y))
            batch_counts, xedges, yedges = np.histogram2d(x, y, nbins, range)
            if counts is None:
                # float64 because histogram2d returns float64
                counts = np.zeros((len(xys),) + batch_counts.shape)
            counts[i] += batch_counts

    for cnts in counts:
        cnts *= 1 / cnts.sum()
    return counts, xedges, yedges


def compute_channel_ranges(
        dataset: Iterable,
        padding_fraction: float = 0.0) -> np.ndarray:
    """Compute min and max for each channel separately.

    Arguments:
        dataset: iterable of arrays of shape (num_channels, state_size)
        padding_fraction: (optional, float) padding to apply to ranges,
                          stretched min and max outwards by
                          (max - min) * padding_fraction

    Output:
        ranges: array of shape (num_channels, 2)
    """
    def sanitize(x):
        if isinstance(x, tuple) and len(x) == 2:
            x = x[0]  # Torch returns (min, argmin)...
        return adaled.to_numpy(x)

    vmin = np.inf
    vmax = -np.inf
    for item in dataset:
        vmin = np.minimum(vmin, sanitize(item.min(axis=1)))
        vmax = np.maximum(vmax, sanitize(item.max(axis=1)))

    if padding_fraction:
        padding = padding_fraction * (vmax - vmin)
        vmin -= padding
        vmax += padding

    return np.stack([vmin, vmax], axis=1)


def compute_1d_channel_stats(
        batches: Iterable[Sequence[np.ndarray]],
        num_bins: int,
        ranges: Sequence[Tuple[float, float]],
        verbose: bool = True) -> Channels1DStats:
    """Compute various statistics on batched multi-channel 1D data.

    Arguments:
        batches: list of batches (batch_size, num_channels, state_size)
        num_bins: number of bins, same for all channels
        ranges: list of (min, max) pairs, one for each channel
    """
    ranges = np.array(ranges, dtype=np.float64)  # Copy.
    histograms = None
    total_sum = None
    total_elements = 0

    vmin = ranges[:, 0, np.newaxis]
    vmax = ranges[:, 1, np.newaxis]
    inv_extent = num_bins / (vmax - vmin)

    for batch in (tqdm.tqdm(batches) if verbose else batches):
        batch = adaled.to_numpy(batch)
        if histograms is None:
            # batch_size = len(batch)
            num_channels = len(batch[0])
            state_size = len(batch[0][0])
            assert ranges.shape == (num_channels, 2), ranges.shape
            histograms = np.zeros(num_channels * state_size * num_bins + 1,
                                 dtype=np.int64)
            offsets = num_bins * np.arange(num_channels * state_size) \
                                 .reshape(num_channels, state_size)
            total_sum = np.zeros((num_channels, state_size))

        total_sum += batch.sum(axis=0)
        total_elements += len(batch)

        # https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
        idx: np.ndarray = np.floor(inv_extent * (batch - vmin))
        idx = idx.astype(np.int32)
        mask = idx < 0
        mask |= idx >= num_bins

        idx += offsets
        idx[mask] = len(histograms) - 1

        histograms += np.bincount(idx.ravel(), minlength=len(histograms))

    means = total_sum / total_elements
    histograms = histograms[:-1].reshape(num_channels, state_size, num_bins)
    bins = np.linspace(vmin[:, 0], vmax[:, 0], num_bins + 1).T
    assert bins.shape == (num_channels, num_bins + 1), \
            (bins.shape, num_channels, num_bins + 1)
    return Channels1DStats(means, histograms, ranges, bins)


def pick_num_of_2d_histogram_bins(N: int, rule: str = 'rice'):
    """Pick a number of bins per dimension for a 2D histogram."""
    # TODO: References for these two rules?
    if rule == 'sturges':
        return int(1 + np.log2(N))
    elif rule == 'rice':
        return max(2, int(2 * N ** (1. / 3)))
    else:
        raise ValueError(f"unknown rule: {rule}")


def postprocess_multichannel_1d(
        transformer,
        dataset: datasets.TrajectoryDataset,
        num_histogram_bins: int = 20,
        ranges: Optional[Sequence[Tuple[float, float]]] = None,
        batch_size: int = 32,
        output_path: str = 'postprocessed_data_1d.pt') -> PostprocessedData1D:
    """Compute various statistics for 1D multi-channel systems."""

    def back_and_forth(batch):
        batch = transformer.transform(batch)
        batch = transformer.inverse_transform(batch)
        return batch

    states = dataset.as_states(('trajectory', 'x'))

    if ranges is None:
        # Add 5% padding by default, used because transformed values may be
        # outside of the original dataset ranges.
        ranges = compute_channel_ranges(states, padding_fraction=0.05)
        print("Using channel ranges:")
        print("   min: ", ranges[:, 0].tolist())
        print("   max: ", ranges[:, 1].tolist())

    print(f"Postprocessing dataset of {len(states)} states...")
    base_dataloader = adaled.DataLoader(states, batch_size=batch_size, shuffle=False)
    original = compute_1d_channel_stats(base_dataloader, num_histogram_bins, ranges)

    print(f"Processing transformed dataset...")
    with torch.no_grad(), adaled.set_train(transformer.model, mode=False):
        dataloader = adaled.WrappedDataLoader(base_dataloader, back_and_forth)
        reconstructed = compute_1d_channel_stats(
                dataloader, num_histogram_bins, ranges)

    data = PostprocessedData1D(
            original, reconstructed, len(dataset.as_trajectories()),
            len(states))
    if output_path:
        io_.save(data, output_path)
        print(f"Done, stored to: {output_path}")
    else:
        print("Done!")
    return data
