from adaled.backends import cmap, get_backend

import numpy as np

from typing import Any, Iterable, Optional

__all__ = ['join_sequences']

def join_sequences(
        sequences: Iterable[Any],
        *,
        gap: Optional[Any] = None) -> Any:
    """Join/concatenate multiple sequences with an optional gap between each
    two consecutive sequences.

    The function is optimized for the case where `sequences` is itself an
    array, tensor or a `TensorCollection`.

    Arguments:
        sequences: (a sequence of arrays, tensors or TensorCollections)
        gap: (optional, scalar) optional gap to add between consecutive
             sequences

    Output: (array, tensor or TensorCollection) concatenated sequences
    """
    try:
        sequences.shape  # Do trajectories have same length?
    except:
        pass
    else:
        return _join_by_reshaping(sequences, gap)

    return _join_list_of_sequences(sequences, gap)


def _join_by_reshaping(sequences, gap):
    """Concatenation where all sequences have same length."""
    if gap is None:
        # No nans needed? then just reshape.
        return cmap(lambda x: x.reshape(-1, *x.shape[2:]), sequences)

    # Nans needed, first allocate sufficiently large buffers, copy data
    num_sequences = len(sequences)
    sequence_length = len(sequences[0])
    total_length = num_sequences * (sequence_length + 1)

    def join(x):
        backend = get_backend(x)
        out = backend.empty((total_length,) + x.shape[2:], dtype=x.dtype)
        reshaped = out.reshape(num_sequences, sequence_length + 1, *x.shape[2:])
        reshaped[:, :sequence_length] = x
        reshaped[:, sequence_length] = gap
        return out

    out = cmap(join, sequences)
    return out[:-1]  # Cut last np.nan.


def _join_list_of_sequences(sequences, gap):
    # First allocate, and ensure that sequences are collection-like.
    shape = sequences[0].shape
    total_length = 0
    for sequence in sequences:
        # TODO: implement the assertion for TensorCollection
        # assert sequence.shape[1:] == shape[1:], \
        #         "all sequences must have the same element shape"
        total_length += len(sequence)

    if gap is not None:
        total_length += len(sequences)

    def allocate(x):
        return get_backend(x).empty(
                (total_length,) + x.shape[1:], dtype=x.dtype)

    out = cmap(allocate, sequences[0])

    # Copy one sequence at a time.
    offset = 0
    for sequence in sequences:
        out[offset : offset + len(sequence)] = sequence
        offset += len(sequence)
        if gap is not None:
            out[offset] = gap
            offset += 1

    if gap is not None:
        out = out[:-1]
    return out


def masked_gaussian_filter1d(
        a: np.ndarray,
        sigma: float,
        *,
        missing_mask: Optional[np.ndarray] = None,
        set_nan: bool = False,
        out: Optional[np.ndarray] = None,
        **kwargs):
    """1D Gaussian filter for data with gaps, i.e. nans."""
    # https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291#36307291
    a = np.asarray(a)
    from scipy.ndimage import gaussian_filter1d  # Lazily import scipy.
    if missing_mask is None:
        missing_mask = np.isnan(a)
    if out is None:
        out = a.copy()
    elif out is not a:
        out[:] = a
    out[missing_mask] = 0
    gaussian_filter1d(out, sigma, output=out, **kwargs)
    w = np.ones_like(a)
    w[missing_mask] = 0
    w = gaussian_filter1d(w, sigma, **kwargs)
    with np.errstate(divide='ignore', invalid='ignore'):
        out /= w

    if set_nan:
        out[missing_mask] = np.nan
    return out


def rolling_average(a: np.ndarray, n: int):
    """Compute rolling average, averaging last `n` elements.

    >>> rolling_average([10., 20., 30., 50., 60., 70.], 4)
    [10.  15.  20.  27.5  40.  52.5]
    """
    # Based on:
    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy/14314054#14314054

    # If a NaN appears in the array, all values after it will be NaNs as well.
    # This can be avoided by constructing a rolling stride and summing up each
    # row separately. See here:
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    # https://github.com/pydata/bottleneck
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        a = np.cumsum(a)
    else:
        a = np.cumsum(a, dtype=np.float64)
    if len(a) <= n:
        a /= np.arange(1, len(a) + 1)
        return a
    a[n:] = a[n:] - a[:-n]  # Not equivalent to a[n:] -= a[:-n]!
    a[:n - 1] /= np.arange(1, n)
    a[n - 1:] /= n
    return a
