from adaled.backends import TensorCollection
from adaled.nn.loss import weighted_mse_losses, weighted_mean_l1_losses
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.transformers.multiresolution import Multiresolution
from .config import CombinedConfigBase, LayerLossConfig, MicroConfigBase
from .micro import CUPTensorCollection, MicroStateHelper, \
        AutoencoderReconstructionHelper
from .utils_2d import compute_divergence_2d_total_l1_losses, \
        compute_derivatives_2d_no_boundary, \
        compute_vorticity_2d_no_boundary, stream_function_to_velocity
from .utils_3d import curl
import adaled

import numpy as np
import torch

from typing import List, Union

_Array = Union[np.ndarray, torch.Tensor]

class ConcatVorticity2DLayer(torch.nn.Module):
    def __init__(self, h: float):
        super().__init__()
        self.h = h

    def forward(self, v: torch.Tensor):
        shape = v.shape
        assert not v.requires_grad
        assert shape[1] == 2
        out = torch.empty((shape[0], 3) + shape[2:], dtype=v.dtype, device=v.device)
        out[:, :2] = v
        out[:, 2, 1:-1, 1:-1] = compute_vorticity_2d_no_boundary(v, self.h)
        return out


class ConcatVorticity3DLayer(torch.nn.Module):
    def __init__(self, h: float):
        super().__init__()
        self.h = h

    def forward(self, v: torch.Tensor):
        shape = v.shape
        assert not v.requires_grad
        assert shape[1] == 3
        out = torch.empty((shape[0], 6) + shape[2:], dtype=v.dtype, device=v.device)
        out[:, :3] = v
        curl(v, self.h, out=out[:, 3:, 1:-1, 1:-1, 1:-1])
        return out


class VorticityLayer(torch.nn.Module):
    """Compute vorticity from the 2D or 3D velocity."""
    def __init__(self, micro_config: MicroConfigBase):
        super().__init__()
        self.hs = micro_config.compute_hs()
        self.ndim = len(micro_config.cells)

        if self.ndim == 2:
            self.func = compute_vorticity_2d_no_boundary
        elif self.ndim == 3:
            self.func = curl
        else:
            raise NotImplementedError()  # unreachable

    def forward(self, v: TensorCollection) -> TensorCollection:
        prefix = 'layer'

        def to_vorticity(key, array):
            layer = int(key[-1][len(prefix):])
            return self.func(array, self.hs[layer])

        return v.named_map(to_vorticity)


class LayerLoss:
    """Computes v, vorticity and divergence loss for each layer."""
    def __init__(self,
                 config: CombinedConfigBase,
                 mr: Multiresolution,
                 is_macro: bool = True,
                 unweighted: bool = False):
        """
        Arguments:
            is_macro: (bool) True if output data is going to be the macro
                    reconstruction, False for micro data. Defaults to True.
            unweighted: (bool) if enabled, all weight factors will be set to 1
                    (i.e. computation is performed for validation/stats
                    purposes, not for training)
        """
        self.config = config.autoencoder.loss
        if unweighted:
            self.config = self.config.unweighted()

        self.micro_helper = MicroStateHelper(config.micro)
        self.macro_helper = AutoencoderReconstructionHelper(config.micro)
        self.is_macro = is_macro

        if config.micro.predict_pressure:
            assert config.micro.pressure_in_state, \
                    "cannot predict pressure loss if there is nothing to compare it with"

        self.num_layers = len(config.micro.multiresolution)
        self.mr = mr
        self.ndim = mr.layers[0].ndim

        self.div_factors = []
        for layer, h in zip(mr.layers, self.macro_helper.hs):
            assert layer.stride[0] == layer.stride[1]
            num_cells = np.prod([s - 2 for s in layer.downscaled_shape])
            # Store 1/num_cells to avoid multiplying later.
            self.div_factors.append(self.config.divergence_weight * 0.5 / h / num_cells)

        for layer in mr.layers:
            assert all(s == layer.stride[0] for s in layer.stride), \
                   "vorticity and divergence computation assumes cells are squares"

        layer_loss_weights = config.compute_multiresolution_weights(mr)
        self.layer_loss_weights = [
            torch.tensor(layer) if layer is not None else layer
            for layer in layer_loss_weights
        ]

    def __call__(self,
                 outputs: TensorCollection,
                 targets: TensorCollection) -> TensorCollection:
        out = {}
        for layer in range(self.num_layers):
            key = f'layer{layer}'
            out[key] = self.layer_loss(layer, outputs[key], targets[key])
        return TensorCollection(out)

    def layer_loss(self, layer: int, output: _Array, target: _Array) -> TensorCollection:
        # as_tensor needed for weighted_mse_losses. Does not copy data.
        output = torch.as_tensor(output)
        target = torch.as_tensor(target)
        weight = self.layer_loss_weights[layer]

        if self.is_macro:
            output_v = self.macro_helper.layer_to_velocity(layer, output)
        else:
            output_v = self.micro_helper.layer_to_velocity(output)
        target_v = self.micro_helper.layer_to_velocity(target)

        losses = {}
        losses['v'] = weighted_mse_losses(
                output_v, target_v, weight,
                relative=self.config.relative_loss,
                eps=self.config.relative_velocity_eps)

        factor = self.config.vorticity_weight
        if factor:
            if self.is_macro:
                output_vort = self.macro_helper.v_to_vorticity_no_boundary(layer, output_v)
            else:
                output_vort = self.micro_helper.layer_to_vorticity_no_boundary(layer, output)
            target_vort = self.micro_helper.layer_to_vorticity_no_boundary(layer, target)
            losses['vorticity'] = factor * weighted_mean_l1_losses(
                    output_vort, target_vort,
                    weight[(slice(1, -1),) * self.ndim] if weight is not None else None,
                    relative=self.config.relative_loss,
                    eps=self.config.relative_vorticity_eps)
            del output_vort  # Release memory.
            del target_vort

        factor = self.config.derivatives_weight
        if factor:
            if self.ndim == 3:
                raise NotImplementedError()
            h = self.macro_helper.hs[layer]
            output_der = compute_derivatives_2d_no_boundary(output_v, h)
            target_der = compute_derivatives_2d_no_boundary(target_v, h)
            losses['derivatives'] = factor * weighted_mse_losses(
                    output_der, target_der,
                    weight[(slice(1, -1),) * self.ndim] if weight is not None else None,
                    relative=self.config.relative_loss,
                    eps=self.config.relative_derivatives_eps)

        factor = self.div_factors[layer]
        if factor:
            if self.config.relative_loss:
                # Note: if implemented, update LayerLossConfig.unweighted.
                raise NotImplementedError("relative loss for divergence not implemented")
            if self.ndim == 2:
                losses['divergence'] = \
                        factor * compute_divergence_2d_total_l1_losses(output, weight)
            else:
                assert self.ndim == 3
                raise NotImplementedError("3D divergence L1 loss not implemented")

        factor = self.config.pressure_weight
        if factor:
            if self.is_macro:
                output_p = self.macro_helper.layer_to_pressure(output)
            else:
                output_p = self.micro_helper.layer_to_pressure(output)
            target_p = self.micro_helper.layer_to_pressure(target)
            if output_p is not None and target_p is not None:
                losses['pressure'] = factor * weighted_mse_losses(
                        output_p, target_p, weight,
                        relative=self.config.relative_loss,
                        eps=self.config.relative_pressure_eps)

        if len(losses.keys()) < 4:
            zeros = torch.zeros(len(output))
            losses.setdefault('vorticity', zeros)
            losses.setdefault('derivatives', zeros)
            losses.setdefault('divergence', zeros)
            losses.setdefault('pressure', zeros)

        return TensorCollection(losses)

    def get_zero_loss(self, backend) -> TensorCollection:
        zero = backend.zeros(())
        empty = {
            'v': zero,
            'vorticity': zero,
            'derivatives': zero,
            'divergence': zero,
            'pressure': zero,
        }
        return TensorCollection({f'layer{k}': empty for k in range(self.num_layers)})


class ReconstructionLoss:
    """Wraps the layer losses {'layer0': ..., ...} into
    {'layers': {'layer0': ..., ...}}. Inherit to add further losses."""
    def __init__(
            self,
            config: CombinedConfigBase,
            mr: Multiresolution,
            is_macro: bool = True,
            unweighted: bool = False):
        self.layer_loss = LayerLoss(config, mr, is_macro=is_macro, unweighted=unweighted)

    def __call__(self, outputs: TensorCollection, targets: TensorCollection) \
            -> TensorCollection:
        layer_loss = self.layer_loss(outputs['layers'], targets['layers'])
        return TensorCollection(layers=layer_loss)

    def get_zero_loss(self, backend) -> TensorCollection:
        return TensorCollection(layers=self.layer_loss.get_zero_loss(backend))


class FullResolutionComparisonError:
    """Compute comparison error based on the full resolution (rebuilt) image.
    The error is the relative velocity MSE, with no epsilon in the
    denominator."""
    def __init__(
            self,
            config: CombinedConfigBase,
            mr: Multiresolution):
        self.micro_helper = MicroStateHelper(config.micro)
        self.macro_helper = AutoencoderReconstructionHelper(config.micro)
        self.config = config
        self.mr = mr

        # If the velocity in the obstacle is hardcoded, do not include it in
        # the comparison error. Alternatively, we could modify the output of
        # the autoencoder, but that's not trivial either. The mask should be
        # applied on velocity and vorticity fields only after the vorticity has
        # been computed. (Namely, we share the same mask for both fields, and
        # vorticity is computed as a derivative of the velocity.)
        if config.micro.enforce_obstacle_velocity:
            self.weight = (~config.micro.compute_obstacle_interior_mask()).astype(np.float32)
        else:
            self.weight = None

    def __call__(self, output: TensorCollection, target: CUPTensorCollection) \
            -> TensorCollection:
        config = self.config
        target_v = torch.from_numpy(
                target.full_resolution_state[:, :self.mr.ndim, ...])

        output = output['layers']
        output_v = []
        for i in range(len(config.micro.multiresolution)):
            v = self.macro_helper.layer_to_velocity(i, output[f'layer{i}'])
            v = v.cpu() if hasattr(v, 'cpu') else torch.from_numpy(v)
            output_v.append(v)
        output_v = self.mr.rebuild_unsafe(output_v)

        error_v = weighted_mse_losses(
                output_v, target_v, self.weight, relative=True, eps=0.0)
        return TensorCollection(v=error_v)

    def get_zero_loss(self, backend) -> TensorCollection:
        return TensorCollection(v=backend.zeros(()))
