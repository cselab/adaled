"""
Utility methods operable on tensors/arrays and TensorCollections.
"""

from adaled.backends.soa import TensorCollection
import adaled.backends as backends

from typing import Optional, Tuple, Union

_DictLike = (dict, TensorCollection)
_ShapeLike = Union[int, Tuple[int, ...]]

def extended_emptylike(
        iterable,
        shape: _ShapeLike,
        axis: int = 0,
        fill: Optional[float] = None):
    """Create an empty collection-like object of the same structure as
    `iterable`, with dimensions of shape `shape` added to each tensor.

    Useful for pre-allocating memory for concatenation.

    Arguments:
        shape: (int or tuple of ints) dimensions to add
        axis: (int) starting dimension, can be negative
        fill: (optional, float) optionally fill with the given value

    >>> emptylike_extended(TensorCollection(a=np.zeros(2, 3, 4), 10, axis=1))
    TensorCollection(a=np.empty(2, 10, 3, 4))

    >>> emptylike_extended(TensorCollection(a=np.zeros(2, 3, 4), (10, 11), axis=1))
    TensorCollection(a=np.empty(2, 10, 11, 3, 4))
    """
    if isinstance(shape, int):
        shape = (shape,)

    def allocate_with_extra_dim(x):
        ax = axis if axis >= 0 else axis + x.ndim + 1
        if fill is None:
            return backends.get_backend(x).empty(
                    x.shape[:ax] + shape + x.shape[ax:], dtype=x.dtype)
        else:
            return backends.get_backend(x).full(
                    x.shape[:ax] + shape + x.shape[ax:], fill, dtype=x.dtype)

    return cmap(allocate_with_extra_dim, iterable)


def extended_nanlike(*args, fill: Optional[float] = float('nan'), **kwargs):
    """Shorthand for `extended_emptylike(..., fill=float('nan'))`."""
    return extended_emptylike(*args, fill=fill, **kwargs)


def cforeach(func, *iterables) -> None:
    """Evaluate the function on each tensor-like object in `*iterables`.

    Same as `cmap`, but doesn't output anything.

    Examples:
        >>> cforeach(lambda x: print(x * x), np.array([10, 20, 30]))
        np.array([100, 200, 300])  # Printed.

        >>> cforeach(lambda x: x * x, TensorCollection({'x': np.array([10, 20, 30]})))
        np.array([100, 200, 300])  # Printed.
    """
    is_collection = isinstance(iterables[0], TensorCollection)

    for other in iterables[1:]:
        if is_collection != isinstance(other, TensorCollection):
            raise TypeError("Inconsistent iterable types. Either none "
                            "or all have to be TensorCollections.")

    if is_collection:
        if len(iterables) == 1:
            iterables[0].foreach(func)  # Should be faster than starforeach.
        else:
            TensorCollection.starforeach(func, iterables)
    else:
        func(*iterables)


def cmap(func, *iterables):
    """Evaluate the function on each tensor-like object in `*iterables` and
    return the result with same structure.

    Examples:
        >>> cmap(lambda x: x * x, np.array([10, 20, 30]))
        np.array([100, 200, 300])

        >>> cmap(lambda x: x * x, TensorCollection({'x': np.array([10, 20, 30]})))
        TensorCollection({'x': np.array([100, 200, 300])})
    """
    if isinstance(iterables[0], (dict, TensorCollection)):
        if len(iterables) == 1:
            # Should be faster than starmap.
            return TensorCollection.map(iterables[0], func)
        else:
            return TensorCollection.starmap(func, iterables)
    else:
        return func(*iterables)


def to_cpu(iterable):
    """Copy tensor-like object to CPU memory, if not already there."""
    return cmap(lambda x: x.cpu() if hasattr(x, 'cpu') else x, iterable)
