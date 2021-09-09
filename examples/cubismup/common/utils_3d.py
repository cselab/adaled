from adaled.backends import get_backend

import numpy as np
import torch

from typing import Optional, Union

_Array = Union[np.ndarray, torch.Tensor]

def divergence_3d(v: np.ndarray, h: float, fill: float = np.nan) -> np.ndarray:
    assert v.ndim == 4
    out = np.full_like(v[0, :, :, :], fill)
    s = slice(1, -1)
    center = out[s, s, s]
    center[:] = v[0, s, s, 2:] - v[0, s, s, :-2]  # dvx/dx
    center += v[1, s, 2:, s] - v[1, s, :-2, s]    # dvy/dy
    center += v[2, 2:, s, s] - v[2, :-2, s, s]    # dvz/dz
    center *= 0.5 / h
    return out


def curl(v: _Array, h: float, out: Optional[_Array] = None) -> _Array:
    """Compute curl of vector fields using a 2nd order cell-centered stencil.

    Arguments:
        v: array or tensor of shape (B, 3, Z, Y, X), where B is the batch size
        h: cell size
        out: (optional) target array, for performance reasons it should be
             used only for numpy and not for torch

    Output:
        array or tensor of shape (B, 3, Z - 2, Y - 2, X - 2)
    """
    # 2nd order accurate cell-centered stencil, the same one used in Cubism.

    # The code below could be made even more memory-efficient (especially when
    # `out` is provided).
    # Note that, at the cost of higher memory, (a - b) -= (c - d) should in
    # principle be slightly more accurate than ((a -= b) -= c) += d, assuming
    # a, b, c and d are all similar in value.
    v0 = v[:, 0, :, :, :]
    v1 = v[:, 1, :, :, :]
    v2 = v[:, 2, :, :, :]
    s = slice(1, -1)
    dv2_d1 = v2[:, s, 2:, s] - v2[:, s, :-2, s]
    dv1_d2 = v1[:, 2:, s, s] - v1[:, :-2, s, s]
    tmp0 = dv2_d1
    tmp0 -= dv1_d2  # -= does not break torch gradients.
    del dv1_d2      # Try to save some memory.

    dv0_d2 = v0[:, 2:, s, s] - v0[:, :-2, s, s]
    dv2_d0 = v2[:, s, s, 2:] - v2[:, s, s, :-2]
    tmp1 = dv0_d2
    tmp1 -= dv2_d0
    del dv2_d0

    dv1_d0 = v1[:, s, s, 2:] - v1[:, s, s, :-2]
    dv0_d1 = v0[:, s, 2:, s] - v0[:, s, :-2, s]
    tmp2 = dv1_d0
    tmp2 -= dv0_d1
    del dv0_d1

    if out is None:
        out = get_backend(v).stack([tmp0, tmp1, tmp2], axis=1)
    else:
        out[:, 0] = tmp0
        out[:, 1] = tmp1
        out[:, 2] = tmp2
    out *= 0.5 / h  # *= also does not break torch gradients.
    return out

def curl_optimized(v: np.ndarray, h: float, out: np.ndarray) -> np.ndarray:
    """Optimized but less accurate curl."""
    v0 = v[:, 0, :, :, :]
    v1 = v[:, 1, :, :, :]
    v2 = v[:, 2, :, :, :]
    tmp0 = out[:, 0]
    tmp1 = out[:, 1]
    tmp2 = out[:, 2]
    factor = 0.5 / h

    s = slice(1, -1)
    np.subtract(v2[:, s, 2:, s], v2[:, s, :-2, s], out=tmp0)
    tmp0 -= v1[:, 2:, s, s]
    tmp0 += v1[:, :-2, s, s]
    tmp0 *= factor

    np.subtract(v0[:, 2:, s, s], v0[:, :-2, s, s], out=tmp1)
    tmp1 -= v2[:, s, s, 2:]
    tmp1 += v2[:, s, s, :-2]
    tmp1 *= factor

    np.subtract(v1[:, s, s, 2:], v1[:, s, s, :-2], out=tmp2)
    tmp2 -= v0[:, s, 2:, s]
    tmp2 += v0[:, s, :-2, s]
    tmp2 *= factor
    return out
