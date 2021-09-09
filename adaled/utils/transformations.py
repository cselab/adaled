import adaled

import numpy as np

from typing import Any, Callable, Union

# Maybe rename to Modifiers, to avoid confusion with transformers (AE)?
Transformation = Union[str, Callable[[Any], Any]]

class _Transformations:
    """Transformations applied to data before recoding or processing.

    Should be simple to configure from command line (or through JSON).
    Not to be confused with transformers.
    """
    def identity(x):
        return x

    def float32(x):
        return adaled.cmap(lambda y: adaled.to_numpy(y).astype('float32'), x)


class _CompoundTransformation:
    __slots__ = ('ts',)
    def __init__(self, *ts):
        self.ts = ts

    def __call__(self, x):
        for t in self.ts:
            x = t(x)
        return x


def parse_transformation(t: Transformation):
    if callable(t):
        return t
    if not isinstance(t, str):
        raise TypeError(f"expected str or callable, got {t}")
    out = getattr(_Transformations, t, None)
    if out is None:
        import inspect
        methods = inspect.getmembers(_Transformations, predicate=inspect.isfunction)
        raise ValueError(f"unknown transformation `{t}`, available: "
                         + ", ".join(k for k, v in methods))
    return out
