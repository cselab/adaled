from adaled.backends import get_backend

import numpy as np
import torch

from typing import Optional, Union

_Array = Union[np.ndarray, torch.Tensor]

def compute_divergence_2d(v: np.ndarray, h: float, fill: float = np.nan):
    assert v.ndim == 3
    out = np.full_like(v[0, :, :], fill)
    s = slice(1, -1)
    center = out[s, s]
    center[:] = v[0, s, 2:] - v[0, s, :-2]  # dvx/dx
    center += v[1, 2:, s] - v[1, :-2, s]    # dvy/dy
    center *= 0.5 / h
    return out


def compute_vorticity_2d(v: np.ndarray, h: float, fill: float = np.nan):
    """Compute vorticity (with the 1/h factor), fill with the given fill value
    at the boundary."""
    assert v.ndim == 3
    out = np.full_like(v[0, :, :], fill)
    out[1:-1, 1:-1] = compute_vorticity_2d_no_boundary(v[np.newaxis], h)
    return out


def compute_vorticity_2d_no_boundary(v: _Array, h: float):
    """Compute vorticity everywhere except the first and last row and column,
    without the 1/h factor. Operates on a batch."""
    vx = v[:, 0, :, :]
    vy = v[:, 1, :, :]
    # 2nd order accurate cell-centered stencil, the same one used in Cubism.
    dvy_dx = vy[:, 1:-1, 2:] - vy[:, 1:-1, :-2]
    dvx_dy = vx[:, 2:, 1:-1] - vx[:, :-2, 1:-1]
    tmp = dvy_dx
    tmp -= dvx_dy
    tmp *= 0.5 / h
    return tmp


def compute_derivatives_2d_no_boundary(v: _Array, h: float):
    vx = v[:, 0, :, :]
    vy = v[:, 1, :, :]
    dvx_dx = vx[:, 1:-1, 2:] - vx[:, 1:-1, :-2]
    dvy_dx = vy[:, 1:-1, 2:] - vy[:, 1:-1, :-2]
    dvx_dy = vx[:, 2:, 1:-1] - vx[:, :-2, 1:-1]
    dvy_dy = vy[:, 2:, 1:-1] - vy[:, :-2, 1:-1]
    out = get_backend(v).stack([dvx_dx, dvy_dx, dvx_dy, dvy_dy, dvy_dx - dvx_dy], axis=1)
    out *= 0.5 / h
    return out


def compute_divergence_2d_total_l1_losses(v: torch.Tensor, weight: Optional[torch.Tensor] = None):
    """Compute total sum of absolute values of divergence everywhere apart from
    the boundary, without the (1/2h) factor. Operates on a batch, one loss is
    returned for each batch sample."""
    vx = v[:, 0, :, :]
    vy = v[:, 1, :, :]
    # 2nd order accurate cell-centered stencil, the same used in Cubism.
    dvx_dx = vx[:, 1:-1, 2:] - vx[:, 1:-1, :-2]
    dvy_dy = vy[:, 2:, 1:-1] - vy[:, :-2, 1:-1]
    # tmp = dvx_dx + dvy_dy
    tmp = dvx_dx
    tmp += dvy_dy
    if weight is not None:
        assert weight.shape == v.shape[-2:], (weight.shape, v.shape)
        tmp *= weight[1:-1, 1:-1]
    return torch.abs(tmp).sum((-1, -2))


def stream_function_to_velocity(psi: _Array, h: float):
    """Compute vx = dpsi/dy, vy = -dpsi/dx for a (NY, NX) grid and return a
    (NY - 2, NX - 2) grid, where `psi` is a stream ("potential") function.

    Idea similar to the 3D solution present in the following paper:
        https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth14.pdf
        Embedding hard physical constraints in convolutional neural networks for 3D turbulence
    """
    assert psi.ndim == 4 and psi.shape[1] == 1, psi.shape  # Only one channel.
    vx = psi[:, 0, 2:, 1:-1] - psi[:, 0, :-2, 1:-1]
    vy = psi[:, 0, 1:-1, :-2] - psi[:, 0, 1:-1, 2:]  # Order is intentional.
    out = get_backend(vx).stack([vx, vy], axis=1)
    out *= 1 / h  # Multiplication does not break gradients.
    return out
