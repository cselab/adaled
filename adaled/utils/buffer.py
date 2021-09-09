from adaled.backends import cmap, get_backend, TensorCollection

import numpy as np

__all__  = ['DynamicArray']

class _EmptyUnknownTypeArray:
    """Array of length 0 of otherwise unknown type or shape."""
    __slots__ = ()

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self
        elif isinstance(key, int):
            raise IndexError()
        elif isinstance(key, tuple) and len(key) == 0:
            return self
        else:
            raise TypeError("invalid key for an empty array: " + str(key))


def _expanded_empty(capacity, hint_elements):
    """Allocate storage equivalent to an array of length `capacity` of
    `hint_element`-like objects."""
    def _empty(x):
        return get_backend(x).empty((capacity,) + x.shape[1:], dtype=x.dtype)

    return cmap(_empty, hint_elements)


class DynamicArray:
    """A contiguous array of tensors or collections of uniform type (numpy vs
    pytorch), shape and dtype, with support for efficient appending of new elements.

    The following code demonstrates the typical usage.

    >>> a = adaled.DynamicArray()
    >>> for i in range(6):
    >>>     a.append(np.arange(4) ** i)  # O(N) complexity, not O(N^2)
    >>> a.data
    array([[  1,   1,   1,   1],
           [  0,   1,   2,   3],
           [  0,   1,   4,   9],
           [  0,   1,   8,  27],
           [  0,   1,  16,  81],
           [  0,   1,  32, 243]])
    >>> a.shape
    (6, 4)
    >>> a[2, :]
    array([0, 1, 4, 9])

    Internally, the data is stored in a larger buffer that grows as new
    elements are added. This enables amortized O(N) complexity for N
    consecutive appends, starting from an empty array.

    The data is stored with the same backend (numpy vs pytorch), device, shape
    and dtype as the original data. Asking for the shape of the array before it
    is known will result in an error.

    >>> a = adalad.DynamicArray()
    >>> len(a)
    0
    >>> a.shape
    RuntimeError: shape not yet known

    To specify the shape immediately, we have following two options.

    >>> placeholder = np.zeros((0, 4, 4))  # Start from an empty buffer.
    >>> a1 = adaled.DynamicArray(placeholder)
    >>> a1.shape
    (0, 4, 4)

    >>> element = np.zeros((4, 4))  # One full element already available.
    >>> a2 = adaled.DynamicArray(like=element)
    >>> a2.shape
    (0, 4, 4)
    """
    __slots__ = ('_buffer', '_size')

    def __init__(self, buffer=None, like=None):
        self._buffer = buffer
        self._size = len(buffer) if buffer is not None else 0
        if like is not None:
            self.append(like)
            self.clear()

    @property
    def __array_interface__(self):
        """Metadata for converting to a numpy array.

        https://numpy.org/doc/stable/reference/arrays.interface.html
        """
        return self.data.__array_interface__

    def __getstate__(self):
        """Store only the used part of the buffer."""
        if self._buffer is not None:
            return self.data
        else:
            return ()  # Empty tag.

    def __setstate__(self, data):
        if isinstance(data, tuple):  # Empty tag.
            self._buffer = None
            self._size = 0
        else:
            self._buffer = data
            self._size = len(data)

    def __len__(self):
        """Get the array length."""
        return self._size

    def __getitem__(self, key):
        """Retrieve an element or a slice of the array."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Update an element or a slice of the array."""
        self.data[key] = value

    def __iter__(self):
        if self._buffer is None:
            return iter([])
        return iter(self._buffer[:self._size])

    def append(self, value):
        """Append the value to the array."""
        self._reserve_extend(1, value[np.newaxis] if self._buffer is None else None)
        self._buffer[self._size] = value
        self._size += 1

    def extend(self, values):
        """Append multiple values to the array."""
        k = len(values)
        if k == 0:
            # Ideally we would like to capture the dtype and shape (or dtypes
            # and shapes in case of a TensorCollection), but for now we simply
            # do nothing.
            return
        self._reserve_extend(k, values)
        self._buffer[self._size:self._size + k] = values
        self._size += k

    def clear(self):
        """Clear the array. Remembers the backend, shape and dtype."""
        self._size = 0

    @property
    def data(self):
        """A view of the data.

        If the shape is not key known, a dummy object of length 0 is returned.
        Attempting to retrieve its shape results in an error.
        """
        if self._buffer is None:
            return _EmptyUnknownTypeArray()
        return self._buffer[:self._size]

    @property
    def shape(self):
        """Return the shape of the array."""
        if self._buffer is None:
            raise RuntimeError("shape not yet known")
        return self._buffer[:self._size].shape

    def _reserve_extend(self, k, hint_elements):
        """Extend the buffer to fit ``k`` new elements if needed. Use the given
        value as a reference for the backend, shape and dtype."""
        if self._buffer is None or self._size + k > len(self._buffer):
            self.reserve((3 * self._size) // 2 + k,
                         hint_elements=hint_elements)
        else:
            # Numpy shape is more or less backend-independent.
            # shape = np.shape(hint_elements)
            # assert shape == self._buffer.shape[1:], \
            #        f"Expected shape {self._buffer.shape[1:]}, got {shape}."
            pass

    def reserve(self, capacity, hint_elements=None):
        """Reserve the buffer for ``capacity`` elements.

        The optional ``hint_elements`` argument is used to determine the element
        backend, shape and type.

        >>> a = adaled.DynamicArray()
        >>> placeholder = np.zeros((0, 4, 4))
        >>> a.reserve(100, placeholder)
        >>> a.shape
        (100, 4, 4)
        """
        if self._buffer is not None and capacity <= len(self._buffer):
            return
        old = self._buffer
        if old is not None:
            self._buffer = _expanded_empty(capacity, old)
            self._buffer[:len(old)] = old
        elif hint_elements is not None:
            self._buffer = _expanded_empty(capacity, hint_elements)
        else:
            raise RuntimeError("Cannot determine storage backend, shape or type, "
                               "provide `hint_elements`.")
