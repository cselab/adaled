from adaled.backends import TensorCollection, cmap

import torch.nn.functional as F
import torch

from typing import Any, Callable, Optional, Sequence, Union

def loss_to_losses(loss):
    """Wrap a loss function to compute one loss per each batch item and per
    tensor (in case of a TensorCollection).

    In some sense, it creates a loss function whose behavior is between
    reduction='mean' and reduction='none'.
    """
    def per_item(*items):
        losses = loss(*items, reduction='none')
        return losses.mean(tuple(range(1, items[0].ndim)))

    def inner(*arrays):
        return cmap(per_item, *arrays)

    return inner


l1_losses = loss_to_losses(F.l1_loss)
mse_losses = loss_to_losses(F.mse_loss)
gaussian_nll_losses = loss_to_losses(F.gaussian_nll_loss)


def weighted_mean_l1_losses(computed, expected, weight, *,
                            relative: bool = False, eps: float = 1e-2):
    """Compute per-sample mean weighted L1 loss.

    If `relative` if `False`, computes
        mean(w * |computed - expected|)

    If `relative` is `True`, computes
        mean(w * |computed - expected|) / (mean(w * |expected|) + eps)
    """
    out = F.l1_loss(computed, expected, reduction='none')
    if weight is not None:
        out *= weight
    out = out.mean(tuple(range(1, expected.ndim)))
    if relative:
        tmp = torch.abs(expected)
        if weight is not None:
            tmp *= weight
        out /= tmp.mean(tuple(range(1, expected.ndim))) + eps
    return out


def weighted_mse_losses(computed, expected, weight, *,
                        relative: bool = False, eps: float = 1e-2):
    """Compute per-sample weighted mean square error.

    If `relative` if `False`, computes
        mean(w * (computed - expected)^2)

    If `relative` is `True`, computes
        mean(w * (computed - expected)^2) / (mean(w * expected^2) + eps).
    """
    out = F.mse_loss(computed, expected, reduction='none')
    if weight is not None:
        out *= weight
    out = out.mean(tuple(range(1, expected.ndim)))
    if relative:
        tmp = torch.square(expected)
        if weight is not None:
            tmp *= weight
        out /= tmp.mean(tuple(range(1, expected.ndim))) + eps
    return out


class MSELosses:
    """Return MSE loss per sample."""
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, a, b):
        return mse_losses(a, b)


class GaussianNLLLosses:
    """Return Gaussian NLL loss per sample."""
    def __call__(self, a, b, c):
        return gaussian_nll_losses(a, b, c)


class ProbabilisticLosses:
    """Mixture of gaussian negative log-likelihood loss and MSE.

    If `mse_weight` is non-None, returns a
        TensorCollection(nll=<NLL losses>, mse=<MSE losses>)
    If `mse_weight` is None, returns only the NLL losses.

    Returns one loss per batch item!
    """
    def __init__(self,
                 nll_weight: float = 1.0,
                 mse_weight: Optional[float] = 1.0,
                 detach_mu: bool = True):
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.detach_mu = detach_mu

    def __repr__(self):
        return f"{self.__class__.__name__}(nll_weight={self.nll_weight}, " \
               f"mse_weight={self.mse_weight})"

    def get_zero_loss(self, backend=torch):
        """Return the zero-filled dummy loss, used to help initialize diagnostics."""
        if self.mse_weight is not None:
            return TensorCollection(nll=backend.zeros(()), mse=backend.zeros(()))
        else:
            return backend.zeros(())

    def __call__(self, output, target) -> Union[torch.Tensor, TensorCollection]:
        output_mu = output[..., 0]
        output_sigma2 = output[..., 1]
        # losses = self.nll_weight * gaussian_nll_losses(
        #         output_mu, target, output_sigma2)
        losses = self.nll_weight * gaussian_nll_losses(
                (output_mu.detach() if self.detach_mu else output_mu),
                target, output_sigma2)
        if not self.mse_weight:
            if self.mse_weight is None:
                return losses
            return TensorCollection(nll=losses, mse=torch.zeros(len(output)))

        # Find a way to ensure that the MSE loss is for sure minimized, and not affected by the rest.
        # Maybe keep the MSE loss, and minimize probabilistic loss ONLY for sigma !

        losses_mse = self.mse_weight * mse_losses(output_mu, target)
        return TensorCollection(nll=losses, mse=losses_mse)


class ScaledLosses:
    def __init__(self, parent_loss: Callable, scale: float):
        self.parent = parent_loss
        self.scale = scale

    def __repr__(self):
        return f"{self.__class__.__name__}(parent={self.parent!r}, scale={self.scale})"

    def __call__(self, *args, **kwargs):
        return self.scale * self.parent(*args, **kwargs)
