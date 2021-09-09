import numpy as np

import warnings

def pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return the indices of points that are on the Pareto front of the given
    point cloud, sorted by x.

    A point (x_i, y_i) is on the Parent front if any only if there are no other
    points (x_j, y_j), i != j, for which:
        x_j > x_i and y_j > y_i
    """
    N = len(xs)
    assert N == len(ys), (N, len(ys))
    if N == 0:
        return np.empty(0, dtype=int)

    if np.isnan(xs + ys).any():
        warnings.warn("encountered nan in pareto_front")

    # No idea why, but the columns in lexsort are ordered in reverse.
    order = np.lexsort((ys, xs))

    keep = []
    for i in order:
        while len(keep) > 0 and ys[keep[-1]] <= ys[i]:
            keep.pop()
        keep.append(i)

    return np.array(keep)
