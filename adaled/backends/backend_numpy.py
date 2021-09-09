import adaled.backends as backends

import numpy as np

import warnings

def make_numpy_like_backend(xp):
    """Common backend class for numpy and cupy."""
    class NumpyLikeBackend(backends.Backend):
        module = xp
        array = xp.array
        empty = xp.empty
        empty_like = xp.empty_like
        full = xp.full
        zeros = xp.zeros
        zeros_like = xp.zeros_like

        @staticmethod
        def clone(x):
            return x.copy()

        @staticmethod
        def detach(x):
            return x  # Nothing to do here.

        moveaxis = xp.moveaxis

        cat = xp.concatenate
        repeat_interleave = xp.repeat
        stack = xp.stack

        @staticmethod
        def strict_reshape(old, shape):
            new = old.reshape(shape)
            if new.base is not (old if old.base is None else old.base):
                raise TypeError(
                        f"cannot reshape without copying the array of shape "
                        f"{old.shape} and stride {old.strides} to shape {shape}")
            return new

    return NumpyLikeBackend


class NumpyBackend(make_numpy_like_backend(np)):
    @staticmethod
    def cast_from(x, warn=False):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (float, list, tuple)):
            return np.asarray(x)

        def _convert(y):
            if isinstance(y, np.ndarray):
                return y
            elif hasattr(y, 'detach'):  # torch
                if warn:
                    warnings.warn("converting torch to numpy")
                return y.detach().cpu().numpy()
            elif 'cupy' in str(y.__class__):
                if warn:
                    warnings.warn("converting cupy to numpy")
                import cupy as cp
                return cp.asnumpy(y)
            else:
                raise TypeError("unrecognized type: ", y.__class__)

        return backends.cmap(_convert, x)
