import sys

from adaled.backends.generic import *

# Do not expose mutable variables like `_np` etc.
__all__ = [
    'Backend', 'get_backend', 'make_backend_cast',
    'AutoInputBackendCast',
    'cforeach', 'cmap', 'TensorCollection',
]

class Backend:
    """Common interface to backends like torch, numpy and cupy.

    The interface is based mostly on torch's interface.

    See backend_*.py files for the implementation of the interfaces.
    """
    __slots__ = ()
    module = None

    def cast_from(self, array, warn=False):
        """Convert a given array to the current backend.

        If `warn` is set to True, a warning will be issued if a conversion was
        performed."""
        raise NotImplementedError(self)

    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError()

    array = _not_implemented
    empty = _not_implemented
    empty_like = _not_implemented
    full = _not_implemented
    zeros = _not_implemented
    zeros_like = _not_implemented
    clone = _not_implemented
    detach = _not_implemented

    moveaxis = _not_implemented

    cat = _not_implemented
    repeat_interleave = _not_implemented
    stack = _not_implemented

    # Reshape if possible without copying, otherwise throw an exception.
    strict_reshape = _not_implemented


# Lazily loaded backends.
_np = None
_torch = None
_NumpyBackend = None
_TorchBackend = None

def _load_numpy():
    global _np
    global _NumpyBackend
    import numpy as _np
    from adaled.backends.backend_numpy import NumpyBackend as _NumpyBackend
    assert _NumpyBackend

def _load_torch():
    global _torch
    global _TorchBackend
    import torch as _torch
    from adaled.backends.backend_torch import TorchBackend as _TorchBackend
    assert _TorchBackend

def _check_and_load():
    """Check which backends were imported and load them here."""
    if not _NumpyBackend and 'numpy' in sys.modules:
        _load_numpy()
    if not _TorchBackend and 'torch' in sys.modules:
        _load_torch()


def get_backend(obj):
    """Return the backend of the given object.

    Accepted arguments:
        numpy.ndarray(...)
        numpy.generic (float64 and similar)
        numpy
        adaled.NumpyBackend
        torch.Tensor(...)
        torch
        adaled.TorchBackend
    """
    _check_and_load()

    if _np:
        if isinstance(obj, (_np.ndarray, _np.generic)) \
                or (isinstance(obj, type) and issubclass(obj, _np.ndarray)) \
                or obj is _np \
                or obj is _NumpyBackend:
            assert _NumpyBackend, (_np, 'numpy' in sys.modules)
            return _NumpyBackend
    if _torch:
        if isinstance(obj, _torch.Tensor):
            assert _TorchBackend
            return _TorchBackend(obj.device)

    raise TypeError(f"Cannot determine backend of object `{obj}` of type `{obj.__class__}`.")


def _identity_cast(x):
    return x


# FIXME: Deprecated?
def make_backend_cast(name):
    """Return a function that converts inputs to the given backend."""
    if name is None:
        return _identity_cast
    elif name == 'numpy':
        _load_numpy()
        return _cast_to_numpy
    elif name == 'torch':
        _load_torch()
        return _TorchBackend('cpu').cast_to_torch
    else:
        raise NotImplementedError(f"Unrecognized backend {name}.")


# FIXME: Deprecated?
class AutoInputBackendCast:
    """Mixin that provides a function `self._preprocess_input` for converting
    arbitrary inputs to the given backend."""
    def __init__(self, *args, input_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_backend = input_backend
        self._preprocess_input = None
        self._setup_cast()

    def __getstate__(self):
        out = self.__dict__.copy()
        del out['_preprocess_input']  # Do not serialize a function.
        return out

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._setup_cast()

    def _setup_cast(self):
        self._preprocess_input = make_backend_cast(self._input_backend)
