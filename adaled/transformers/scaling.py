from adaled.backends import TensorCollection, cmap, get_backend
from adaled.transformers.base import Transformer
from adaled.utils.misc import get_global_default_np_dtype
import adaled

import numpy as np
import torch

from dataclasses import dataclass
from typing import Any

class ScalingLayer(torch.nn.Module):
    """Represents a layer :math:`scale * (x + shift)`.

    Args:
        scale (float, array or collection): multiplicative factor
        shift (float, array or collection): offset
    """
    def __init__(self, scale, shift):
        super().__init__()
        # Scale and shift are tensors, not parameters.
        self.scale = cmap(torch.tensor, scale)
        self.shift = cmap(torch.tensor, shift)
        # TODO: Refactor shift to scale_shift.
        self.scale_shift = self.scale * self.shift

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'scale_shift' not in state:
            self.scale_shift = self.scale * self.shift

    def forward(self, x):
        # return self.scale * (x + self.shift)
        try:
            # Computes scale_shift + scale * x in one go, useful when x is huge.
            return torch.addcmul(self.scale_shift, self.scale, x)
        except Exception as e:
            # addcmul is quite sensitive regarding the input, add diagnostics
            # here if it rejects it.
            a, b = self.scale_shift, self.scale
            a_dtype = getattr(a, 'dtype', '??')
            b_dtype = getattr(b, 'dtype', '??')
            x_dtype = getattr(x, 'dtype', '??')
            raise e.__class__(
                    f"addcmul threw an exception for the input of shapes "
                    f"({a.shape}, {b.shape}, {x.shape}) and dtypes "
                    f"({a_dtype}, {b_dtype}, {x_dtype}) "
                    f"for scale*shift={a} and scale={b}")


class Scaling:
    """Defines linear scaling of data in the form `scale * (x + shift)`.

    The scaling can be defined in multiple different ways, by providing
        a) (scale, shift) values
        b) (mean, std): scale and shift computed such that data normalizes
           to mean 0 and std 1.
        c) (min, max): scale and shift computed such that data normalized
           to min -1 and max +1.
    """
    __slots__ = ('shift', 'scale')

    def __init__(self, *, scale=None, shift=None, mean=None, std=None, min=None, max=None):
        if (scale is None) != (shift is None):
            raise TypeError("either neither or both `scale` and `shift` have to be specified")
        if (mean is None) != (std is None):
            raise TypeError("either neither or both `mean` and `std` have to be specified")
        if (min is None) != (max is None):
            raise TypeError("either neither or both `min` and `max` have to be specified")
        if (scale is not None) + (mean is not None) + (min is not None) != 1:
            raise TypeError("expected exactly one scaling specification")

        def _to_numpy_or_float(x):
            """Convert to numpy. Keep scalars as scalars."""
            if isinstance(x, float):
                return x
            elif isinstance(x, TensorCollection):
                return x.map(_to_numpy_or_float)
            elif isinstance(x, dict):
                return cmap(_to_numpy_or_float, x)
            else:
                return np.array(x)

        if mean is not None:
            self.shift = -_to_numpy_or_float(mean)
            self.scale = np.float32(1) / _to_numpy_or_float(std)
        elif min is not None:
            min = _to_numpy_or_float(min)
            max = _to_numpy_or_float(max)
            # Use np.float32 because float32 * float produces float64.
            self.shift = (min + max) * np.float32(-0.5)
            self.scale = np.float32(2) / (max - min)  # Or whatever factor goes here...
        else:
            self.shift = _to_numpy_or_float(shift)
            self.scale = _to_numpy_or_float(scale)

    def __str__(self):
        return f"Scaling(scale={self.scale}, shift={self.shift})"

    def __repr__(self):
        return f"Scaling(scale={self.scale}, shift={self.shift})"

    def __getitem__(self, key: str):
        """Get scaling of a specific item :attr:`key`, assuming the scaling was
        defined for a :class:`TensorCollection`."""
        return Scaling(scale=self.scale[key], shift=self.shift[key])

    def astype(self, dtype):
        """Convert scaling factors and shift to the given numpy type."""
        return Scaling(scale=dtype(self.scale), shift=dtype(self.shift))

    def inversed(self):
        """Return a :class:`Scaling` corresponding to the inverse scaling."""
        # Use np.float32 because int / float32 produces float64.
        return Scaling(scale=np.float32(1) / self.scale,
                       shift=-self.shift * self.scale)

    def to_torch(self, autocast: bool = True) -> ScalingLayer:
        """Convert to :class:`ScalingLayer`."""
        scale = self.scale
        shift = self.shift
        if autocast:
            dtype = get_global_default_np_dtype()
            scale = cmap(dtype, scale)
            shift = cmap(dtype, shift)
        return ScalingLayer(scale, shift)

    def to_transformer(self):
        """Convert to an autoencoder :class:`ScalingTransformer`."""
        return ScalingTransformer(self)

    def asdict(self):
        """Convert to dict. For dataclass's pretty_print."""
        return {'shift': self.shift, 'scale': self.scale}


# TODO: Either remove ScalingTransformer or ScaledAutoencoderTransformer.
class ScalingTransformer(Transformer):
    """Transformer that scales the data according to the given scaling."""

    def __init__(self, scaling: Scaling):
        from adaled.transformers.autoencoders import AutoencoderModule
        self.scaling = scaling.to_torch()
        self.inv_scaling = scaling.inversed().to_torch()
        self.model = AutoencoderModule(self.scaling, self.inv_scaling)

    def transform(self, x):
        return self.scaling(x)

    def inverse_transform(self, z):
        return self.inv_scaling(z)
