import adaled
import numpy as np

from typing import Any, Callable, Dict, Optional, Sequence, Union, Tuple
import copy
import functools
import operator

# TODO: Implement numpy's array mechanism such that e.g. np.log(t) calls
# TensorCollection's log function.

# TODO: Can TensorCollection be implemented as a simple wrapper around dict,
# without wrapping the whole hierarchy with TensorCollections? If so, how would
# shallow copies behave? Would `TensorCollection(t) is t` be True or False for
# a TensorCollection t?

_ArrayLike = Union[np.ndarray, 'torch.Tensor']
_CollectionLike = Union['TensorCollection', _ArrayLike]

class EmptyTensorCollectionError(TypeError):
    pass

class _NoShape:
    def __repr__(self):
        return 'NoShape'

NoShape = _NoShape()


def preprocess_for_tensor_collection(value, default_numpy: bool = False):
    def rec(value):
        if isinstance(value, dict):
            return TensorCollection(
                    {k: rec(v) for k, v in value.items()}, no_parse=True)
        elif default_numpy and isinstance(value, (int, float, list)):
            value = np.array(value)
            if value.dtype is object:
                raise TypeError("too complicated input, handle it manually", value)
        return value

    return rec(value)


def _get_parts(collections: Sequence[_CollectionLike], key: str):
    """Get `key` from each collection."""
    return [c[key] for c in collections]


def _get_parts_with_broadcast(objs: Sequence[_CollectionLike], key: str):
    """Get `key` from each object. If object is a collection, get `obj[key]`,
    otherwise assume broadcasting and return `obj` as is."""
    return [(o[key] if isinstance(o, (dict, TensorCollection)) else o) for o in objs]


def _make_binary_operator(op):
    def operator(*operands):  # (TensorCollection, Any)
        return TensorCollection.starmap(
                op, operands, _get_parts_with_broadcast)
    return functools.wraps(op)(operator)


def _make_reverse_operator(rev_op):
    def operator(*operands):  # (TensorCollection, Any)
        return TensorCollection.starmap(
                rev_op, operands, _get_parts_with_broadcast)
    return functools.wraps(rev_op)(operator)


def _make_assignment_operator(op):
    def operator(*operands):
        TensorCollection.starforeach(
                op, operands, _get_parts_with_broadcast)
        return operands[0]  # self
    return functools.wraps(op)(operator)


class TCShape:
    """Shape of a TensorCollection. Immutable."""
    __slots__ = ('dict',)

    def __new__(cls, d: Union[dict, 'TCShape']):
        if isinstance(d, dict):
            out = super().__new__(cls)
            super().__setattr__(out, 'dict', d)
            return out
        elif isinstance(d, TCShape):
            return d
        # elif isinstance(d, tuple):
        #     return d  # Not a TCShape, probably an ordinary shape.
        else:
            raise TypeError(f"expected dict or TCShape, got {d!r}")

    def __repr__(self):
        return f'{self.__class__.__name__}({self._clean_str()})'

    def _clean_str(self):
        content = ', '.join(f'{k!r}: {v._clean_str() if isinstance(v, TCShape) else v}'
                            for k, v in self.dict.items())
        return f'{{{content}}}'

    def __getitem__(self, key: Union[str, slice, int]) -> Union[tuple, 'TCShape']:
        """Return the shape of a given tensor or subhierarchy or slice all shapes.

        Supports e.g. ``shape['key']``, ``shape[0]`` and ``shape[:2]``.
        """
        if isinstance(key, str):
            key = self.dict[key]  # Reuse variable.
            return TCShape(key) if isinstance(key, dict) else key
        else:
            return self.map(lambda x: x[key])

    def __setattr__(self, key, value):
        raise AttributeError(f"{self.__class__.__name__} is immutable")

    def __eq__(self, other):
        if isinstance(other, TCShape):
            return self.dict == other.dict
        elif isinstance(other, dict):
            return self.dict == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.dict)

    def map(self, func: Callable[[tuple], 'TCShape']) -> 'TCShape':
        """Apply a given function on every element of the hierarchy and return
        the new TCShape of the function outputs."""
        def rec(d):
            return {
                key: (rec(child) if isinstance(child, dict) else func(child))
                for key, child in d.items()
            }

        return TCShape(rec(self.dict))



class TensorCollection:
    """Structure-of-arrays storage for heterogeneous data.

    TensorCollection is a data structure that blends tensors with dictionaries.
    Storage-wise, a collection is a hierarchical dictionary of tensors.
    It provides methods like ``a.items()``, ``a.keys()``, ``a.values()`` and
    provides key access like ``a['x']`` or ``a['x'] = value``.

    Logically, however, a collection is an array.
    The length of a collection ``a`` is determined by the length of the underlying tensors.
    If their lengths are not unique, invoking ``len(a)`` raises a ``TypeError``.
    TensorCollection provides indexing, slicing and iterating just like tensors.

    Multiple mixed keys can be specified at the same time.

    >>> x = np.array([[10, 15, 20], [50, 55, 60]])
    >>> y = np.array([[10, 15, 20, 25], [50, 55, 60, 65]])
    >>> a = TensorCollection(x=x, y=y)
    >>> a
    <TensorCollection of shape {'x': (2, 3), 'y': (2, 4)}>
    >>> a['x']
    array([[10, 15, 20],
           [50, 55, 60]])
    >>> a['x', 0]
    array([10, 15, 20])
    >>> a['x', :, 0]
    array([10, 50])
    >>> a[0]
    <TensorCollection of shape {'x': (3,), 'y': (4,)}>
    >>> a[0, 0, 0]
    IndexError

    >>> b = TensorCollection(x=np.zeros((100, 10)), y=np.zeros((100, 20)))
    >>> b.shape
    TCShape({'x': (100, 10), 'y': (100, 20)})
    >>> len(b)
    100
    >>> c = TensorCollection(x=np.zeros((100, 10)), y=np.zeros((50, 20)))
    >>> len(c)
    TypeError: incompatible tensor lengths, len() not well defined: shape=TCShape({'x': (100, 10), 'y': (50, 20)})

    >>> z = np.zeros((10, 5))
    >>> d = TensorCollection({'first': {'second': z}})
    >>> d['first']
    <TensorCollection of shape {'second': (10, 5)}>
    >>> d.shape
    TCShape({'first': {'second': (10, 5)}})
    >>> d.hierarchy()
    {'first': {'second': None}}


    Tensor collections are intended to be as similar to tensors as possible.
    Thus, they support arithmetic operations like addition, subtraction,
    multiplication and division.

    >>> a = TensorCollection(x=np.arange(5), y=np.arange(3))
    >>> b = TensorCollection(x=np.arange(5), y=np.arange(3))
    >>> (a + b)['x']
    [0 2 4 6 8]
    >>> (a ** 2)['x']
    [0 1 4 9 16]
    """
    __slots__ = ('dict',)

    # https://stackoverflow.com/questions/45947152/custom-class-with-add-to-add-with-numpy-array
    # Ensure that e.g. ndarray + TensorCollection calls TensorCollection.__radd__.
    __array_ufunc__ = None

    def __init__(self, _dict={}, *,
                 default_numpy: bool = False,
                 no_parse: bool = False,
                 **_dict2):
        """Create a TensorCollection.

        The TensorCollection can be created either by providing a dictionary of the data or by passing elements as keyword arguments.

        >>> TensorCollection(a=np.zeros(3), b=np.zeros(5))
        <TensorCollection of shape {'a': (3,), 'b': (5,)}>
        >>> TensorCollection({'a': np.zeros(3), 'b': np.zeros(5)})
        <TensorCollection of shape {'a': (3,), 'b': (5,)}>
        """
        if no_parse:
            assert not _dict2
            self.dict = _dict
        else:
            d = {}
            for key, value in _dict.items():
                d[key] = preprocess_for_tensor_collection(value, default_numpy)
            for key, value in _dict2.items():
                d[key] = preprocess_for_tensor_collection(value, default_numpy)
            self.dict = d

    def __repr__(self):
        return f"<TensorCollection of shape {self.shape._clean_str()}>"

    def describe(self):
        """Print a human-readable formatted overview of the content."""
        def info(x):
            if hasattr(x, 'device'):
                return f"<{x.shape}, {x.dtype}, {x.device}>"
            elif hasattr(x, 'shape'):
                return f"<{x.shape}, {x.dtype}>"
            else:
                return f"<{x.__class__.__name__}>"

        import json
        print(json.dumps(self.map(info).asdict(), indent=4))

    def __format__(self, fmt):
        def rec(x):
            if isinstance(x, TensorCollection):
                return "{" + ", ".join(f"{k}: {rec(v)}" for k, v in x.items()) + "}"
            else:
                return format(x, fmt)
        # return f"TensorCollection({rec(self)})"
        return f"{rec(self)}"

    # Torch-like functions BEGIN.
    def clone(self):
        """Perform a deep copy."""
        return self.map(lambda x: adaled.get_backend(x).clone(x))

    def cpu(self):
        """Copy all arrays to the CPU memory.

        For arrays already on CPU, no copying is performed.
        """
        return self.map(lambda x: x.cpu() if hasattr(x, 'cpu') else x)

    def cpu_numpy(self):
        """Shorthand for .cpu().numpy()."""
        return self.map(lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x)

    def numpy(self):
        """Conver to numpy, assuming the arrays are in CPU memory."""
        return self.map(lambda x: x.numpy() if hasattr(x, 'numpy') else x)

    def detach(self):
        """Evaluate .detach() on each element and return the corresponding
        values in a collection of the same structure."""
        return self.map(lambda x: x.detach())

    def item(self):
        """Evaluate .item() on each element and return the corresponding values
        in a collection of same structure."""
        return self.map(lambda x: x.item())
    # Torch-like functions END.

    def keys(self):
        """Return the top-level hierarchy keys."""
        return self.dict.keys()

    def items(self):
        """Return the top-level hierarchy (key, value) pairs."""
        return self.dict.items()

    def values(self):
        """Return the top-level hierarchy values."""
        return self.dict.values()

    def allitems(self):
        """Return an iterable over all (keys, value) pairs, recursively.

        >>> list(TensorCollection({'x': {'y': y}, 'z': z}).allitems())
        [(('x', 'y'), y), (('z',), z)]
        """
        def rec(keys: Tuple[str, ...], value: dict):
            for k, v in value.items():
                if isinstance(v, TensorCollection):
                    yield from rec(keys + (k,), v.dict)
                else:
                    yield (keys + (k,), v)

        yield from rec((), self.dict)

    def allvalues(self):
        """Return an iterable over all values, recursively.

        >>> list(TensorCollection({'x': {'y': y}, 'z': z}).allvalues())
        [y, z]
        """
        for value in self.dict.values():
            if isinstance(value, TensorCollection):
                yield from value.allvalues()
            else:
                yield value

    def asdict(self):
        """Return the collection as a recursive dictionary.
        Values are left as-is, without copying.

        >>> a = TensorCollection({'a': {'b': np.arange(5)}})
        {'a': {'b': array([0, 1, 2, 3, 4])}}
        """
        return {k: (v.asdict() if isinstance(v, TensorCollection) else v)
                for k, v in self.dict.items()}

    def concat_flatten(self, join='.'):
        """Return a TensorCollection containing all values, recursively,
        flattened to depth 1. The keys are constructed by joining hierarchical
        keys.

        >>> TensorCollection({'x': {'y': np.arange(5)}}).concat_flatten()
        <TensorCollection of shape {'x.y': (5,)}>
        """
        out = {}
        def add(keys, value):
            out[join.join(keys)] = value
        self.named_foreach(add)
        return TensorCollection(out)

    def hierarchy(self) -> Dict[str, Optional[Dict]]:
        """Return the tensor hierarchy as a recursive dictionary of keys.
        Tensors are replaces with ``None``.

        >>> TensorCollection({'x': {'y': y}, 'z': z}).hierarchy()
        {'x': {'y': None}, 'z': None}
        """
        return {k: (v.hierarchy() if isinstance(v, TensorCollection) else None)
                for k, v in self.dict.items()}

    def remove_empty(self):
        """Return a shallow copy with empty hierarchy removed.

        >>> TensorCollection({'x': {'y': {}}}).remove_empty()
        <TensorCollection of shape {}>
        """
        children = {}
        for key, value in self.dict.items():
            if isinstance(value, TensorCollection):
                value = value.remove_empty()
                if len(value.dict.keys()) == 0:
                    continue
            children[key] = value
        return TensorCollection(children, no_parse=True)

    def __contains__(self, key):
        """Check whether the given string or integer key exists.

        If key is a string, returns whether it exists in the (top-level) hierarchy.
        If key is an integer, checks whether it is within bounds of all tensors, i.e. whether ``self[key]`` is well-defined.
        Currently, requires that ``len(self)`` is well-defined.
        This requirement could be lifted if necessary.

        Multikeys not supported.
        Returns ``False`` for any other type.
        """
        if isinstance(key, str):
            return key in self.dict
        elif isinstance(key, int):
            return 0 <= key < len(self)
        else:
            return False

    def __len__(self):
        """Return the length of tensors (not the number of keys in the collection!).

        Raises a ``TypeError`` if not all tensors have the same length.
        If the collection has no tensors, an ``EmptyTensorCollectionError`` is raised.
        """
        out = None
        for value in self.dict.values():
            try:
                l = len(value)
            except EmptyTensorCollectionError:
                continue
            if out is not None and l != out:
                raise TypeError(f"incompatible tensor lengths, "
                                f"len() not well defined: shape={self.shape}")
            out = l
        if out is None:
            raise EmptyTensorCollectionError("len() not defined on an empty TensorCollection")
        return out

    def get(self, key: str, default: Any = None):
        """Equivalent of ``dict.get(key, default)``.

        Returns the tensor or sub-collection ``self[key]`` if it exists.
        Otherwise, returns ``default`` (``None`` by default).
        """
        if isinstance(key, str):
            out = self.dict.get(key)
            if out is None:
                return default
            elif isinstance(out, dict):
                return TensorCollection(out, no_parse=True)
            else:
                return out
        else:
            raise TypeError(f"get() expected a str, got {key} instead")

    def __getitem__(self, key):
        """Get a tensor, a sub-collection or slice the tensors.

        Supports multiple keys.
        """
        # Note: maybe unexpectedly, a[()] does not produce a "shallow" copy.
        # If needed, this can be fixed.
        if not isinstance(key, tuple):
            key = (key,)
        remainder = []
        for k in key:
            if isinstance(k, str):
                if isinstance(self, TensorCollection):
                    self = self.dict[k]
                else:
                    raise KeyError(f"attempting to access item {k!r} of an array")
            else:
                remainder.append(k)
        if not remainder:
            return self
        remainder = tuple(remainder)
        if isinstance(self, TensorCollection):
            return self.map(lambda x: x[remainder])
        else:
            return self[remainder]  # Array-like.

    # Unary operators.
    def __neg__(self):
        return self.map(operator.neg)

    def __pos__(self):
        return self.map(operator.pos)

    def __abs__(self):
        return self.map(operator.abs)

    # Comparison operators.
    # TODO

    # Binary operators.
    # TODO: bitwise operators
    __add__ = _make_binary_operator(operator.add)
    __sub__ = _make_binary_operator(operator.sub)
    __mul__ = _make_binary_operator(operator.mul)
    __matmul__ = _make_binary_operator(operator.matmul)
    __mod__ = _make_binary_operator(operator.mod)
    __floordiv__ = _make_binary_operator(operator.floordiv)
    __truediv__ = _make_binary_operator(operator.truediv)
    __pow__ = _make_binary_operator(operator.pow)

    __radd__ = _make_reverse_operator(lambda a, b: b + a)
    __rsub__ = _make_reverse_operator(lambda a, b: b - a)
    __rmul__ = _make_reverse_operator(lambda a, b: b * a)
    __rmatmul__ = _make_reverse_operator(lambda a, b: b @ a)
    __rmod__ = _make_reverse_operator(lambda a, b: b % a)
    __rfloordiv__ = _make_reverse_operator(lambda a, b: b // a)
    __rtruediv__ = _make_reverse_operator(lambda a, b: b / a)
    __rpow__ = _make_reverse_operator(lambda a, b: b ** a)

    # Assignment binary operators.
    # TODO: bitwise operators
    __iadd__ = _make_assignment_operator(operator.iadd)
    __isub__ = _make_assignment_operator(operator.isub)
    __imul__ = _make_assignment_operator(operator.imul)
    __imatmul__ = _make_assignment_operator(operator.imatmul)
    __imod__ = _make_assignment_operator(operator.imod)
    __ifloordiv__ = _make_assignment_operator(operator.ifloordiv)
    __itruediv__ = _make_assignment_operator(operator.itruediv)
    __ipow__ = _make_assignment_operator(operator.ipow)

    # Torch and numpy-like functions BEGIN.
    def sum(self, *args, **kwargs):
        """Invoke ``x.sum()`` on each tensor ``x`` and return a collection of results."""
        return self.map(lambda x: x.sum(*args, **kwargs))

    def mean(self, *args, **kwargs):
        """Invoke ``x.mean()`` on each tensor ``x`` and return a collection of results."""
        return self.map(lambda x: x.mean(*args, **kwargs))
    # Torch and numpy-like functions END.

    def pop(self, key):
        """Remove an item from the collection. Return the corresponding value."""
        return self.dict.pop(key)

    def __delitem__(self, key: str):
        """Delete the element matching the given key.

        Multikeys not supported.
        """
        if isinstance(key, str):
            del self.dict[key]
        else:
            raise TypeError(f"expected a str, got {key} instead")

    def __setitem__(self, key, value):
        """Element assignment, ``self[key] = value``.

        If key is a ``str``, the corresponding element is added or replaced with ``value``.
        Otherwise, if ``value`` is a dictionary, its values are recursively assigned to ``self[key]``.
        In that case, the dictionary ``value`` and ``self[key]`` must have the same hierarchy.

        In other cases, each tensor of ``self`` is assigned ``value``.
        Useful for e.g. ``a[:] = np.nan``.

        Multikey assignment (``a['x', 'y'] = value``) currently supported.
        """
        value = preprocess_for_tensor_collection(value)
        if isinstance(key, str):
            self.dict[key] = value
        elif isinstance(value, (dict, TensorCollection)):
            if value.keys() != self.dict.keys():
                raise TypeError(f"assigning a tensor collection works only "
                                f"when keys agree (lhs={set(self.dict.keys())}, "
                                f"rhs={set(value.keys())}")
            for k, v in self.dict.items():
                v[key] = value[k]
        else:
            for v in self.dict.values():
                v[key] = value

    def named_foreach(self, fn):
        """Invoke the given function for each tensor, recursively.

        The function is passed two arguments: keys (tuple of strings) and the value (the tensor).

        >>> a = TensorCollection({'x': {'y': np.arange(5)}})
        >>> a.named_foreach(lambda keys, value: print(keys, value))
        ('x', 'y') [0 1 2 3 4]
        """
        def rec(keys, x):
            for key, value in x.items():
                if isinstance(value, TensorCollection):
                    rec(keys + (key,), value)
                else:
                    fn(keys + (key,), value)

        rec((), self.dict)

    def foreach(self, fn):
        """Call the given function for each tensor, recursively.

        >>> a = TensorCollection({'x': {'y': np.arange(5)}})
        >>> a.foreach(lambda value: print(value))
        [0 1 2 3 4]
        """
        for value in self.dict.values():
            if isinstance(value, TensorCollection):
                value.foreach(fn)
            else:
                fn(value)

    @staticmethod
    def starforeach(fn, collections, parts_fn=_get_parts):
        for key, value in collections[0].items():
            parts = parts_fn(collections, key)
            if isinstance(value, TensorCollection):
                TensorCollection.starforeach(fn, parts, parts_fn)
            else:
                fn(*parts)

    def named_map(self, fn):
        """Same as ``map``, but also passes keys tuple as the first argument to
        the given function. See also ``named_foreach``.

        Furthermore, if the function returns a ``None``, the corresponding item
        is excluded from the output collection.
        """
        def rec(keys, x):
            out = {}
            for key, value in x.items():
                if isinstance(value, TensorCollection):
                    value = rec(keys + (key,), value)
                else:
                    value = fn(keys + (key,), value)
                if value is not None:
                    out[key] = value
            return TensorCollection(out)

        return rec((), self.dict)

    def map(self, fn):
        """Invoke the given function for each tensor and create a new
        TensorCollection with the function output.

        >>> a = TensorCollection(x1=np.arange(3), x2=np.arange(5))
        >>> a.map(lambda x: np.concatenate([x, x]))
        <TensorCollection of shape {'x1': (6,), 'x2': (10,)}>
        """
        out = {}
        for key, value in self.items():
            if isinstance(value, TensorCollection):
                value = value.map(fn)
            else:
                value = fn(value)
            out[key] = value
        return TensorCollection(out)

    @property
    def shape(self) -> TCShape:
        """The hierarchical shape of the collection."""
        return TCShape({key: getattr(value, 'shape', NoShape)
                        for key, value in self.dict.items()})

    @staticmethod
    def multimap(fn, collections, *, parts_fn=_get_parts):
        """Iterate over multiple collections, invoke a given function for each
        tensor tuple and collect the results into a new collection.

        >>> u = TensorCollection(x=np.arange(10), y=np.arange(20))
        >>> v = TensorCollection(x=np.arange(11), y=np.arange(22))
        >>> TensorCollection.multimap(lambda uv: np.concatenate(uv), [u, v])
        <TensorCollection of shape {'x': (21,), 'y': (42,)}>
        """
        out = {}
        elem = collections[0]
        for key, value in elem.items():
            parts = parts_fn(collections, key)
            if isinstance(value, TensorCollection):
                out[key] = TensorCollection.multimap(fn, parts)
            else:
                out[key] = fn(parts)
        return TensorCollection(out)

    @staticmethod
    def starmap(fn, collections, parts_fn=_get_parts):
        """Equivalent of ``multimap``, but the tensors are expanded into
        arguments when passed to the function.

        >>> u = TensorCollection(x=np.arange(10), y=np.arange(20))
        >>> v = TensorCollection(x=np.arange(11), y=np.arange(22))
        >>> TensorCollection.starmap(lambda a, b: np.concatenate([a, b]), [u, v])
        <TensorCollection of shape {'x': (21,), 'y': (42,)}>
        """
        out = {}
        elem = collections[0]
        for key, value in elem.items():
            parts = parts_fn(collections, key)
            if isinstance(value, TensorCollection):
                out[key] = TensorCollection.starmap(fn, parts, parts_fn)
            else:
                out[key] = fn(*parts)
        return TensorCollection(out)


_CollectionLike = Union[Dict[str, Any], TensorCollection]
