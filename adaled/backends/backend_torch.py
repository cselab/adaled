import adaled.backends as backends

import torch

import functools
import warnings

_cache = {}

def _add_device_kwarg(fn):
    def inner(backend, *args, **kwargs):
        kwargs.setdefault('device', backend.device)
        return fn(*args, **kwargs)

    return functools.wraps(fn)(inner)

def _ignore_self(fn):
    def inner(backend, *args, **kwargs):
        return fn(*args, **kwargs)

    return functools.wraps(fn)(inner)


def _check_device(device):
    """Automatically fall back to CPU if CUDA not available."""
    if 'cuda' in str(device) and not torch.cuda.is_available():
        return 'cpu'
    else:
        return device


class TorchBackend(backends.Backend):
    __slots__ = ('device',)
    module = torch

    def __new__(cls, device):
        try:
            return _cache[device]
        except KeyError:
            pass
        _cache[device] = backend = super().__new__(cls)
        return backend

    def __init__(self, device):
        self.device = device

    def __getnewargs__(self):
        return (self.device,)

    def __getstate__(self):
        return self.device

    def __setstate__(self, device):
        self.device = _check_device(device)

    def cast_from(self, x, warn=False):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device)
        if isinstance(x, (float, list, tuple)):
            return torch.as_tensor(x, device=self.device)
        if warn:
            warnings.warn(f"Conversion from {x.__class__} to torch.Tensor!", stacklevel=2)
        return backends.cmap(lambda x: torch.as_tensor(x, device=self.device), x)

    def empty_like(self, x, *, shape=None, dtype=None, layout=None, device=None):
        """Add to existing `torch.empty_like` an option to customize shape,
        while keeping other attributes the same."""
        if shape is None:
            shape = x.shape
        if dtype is None:
            dtype = x.dtype
        if layout is None:
            layout = x.layout
        if device is None:
            device = x.device
        return torch.empty(shape, dtype=dtype, layout=layout, device=device)

    array = _add_device_kwarg(torch.tensor)
    empty = _add_device_kwarg(torch.empty)
    full = _add_device_kwarg(torch.full)
    zeros = _add_device_kwarg(torch.zeros)
    zeros_like = _ignore_self(torch.zeros_like)
    clone = _ignore_self(torch.clone)
    detach = _ignore_self(torch.detach)

    moveaxis = _ignore_self(torch.moveaxis)

    cat = _ignore_self(torch.cat)
    repeat_interleave = _ignore_self(torch.repeat_interleave)
    stack = _ignore_self(torch.stack)

    @staticmethod
    def strict_reshape(tensor, shape):
        return tensor.view(shape)
