from adaled.backends import get_backend
import numpy as np

def random_subsequences(data, length, axis=1):
    """Given a (N, K, ...) matrix, return a matrix (N, K', ...) with K' <= K,
    such that a random contiguous part of length K' is taken out of the K
    elements.

    Useful for preparing epoch data for training RNNs.

    Arguments:
        length: the desired length of final sequences
        axis: which axis to shorten
    """
    shape = data.shape
    left, curr, right = shape[:axis], shape[axis], shape[axis + 1:]
    assert length <= curr, (length, curr, data.shape)

    size = 1
    for dim in left:
        size *= dim
    offsets = np.random.randint(0, curr - length + 1, size)
    out = get_backend(data).empty((*left, length, *right))

    for k, ii in enumerate(np.ndindex(*left)):
        out[ii] = data[(*ii, slice(offsets[k], offsets[k] + length))]

    return out
