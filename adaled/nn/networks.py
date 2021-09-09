# Created by: Vlachas Pantelis, Ivica Kicic, CSE-lab, ETH Zurich

from adaled.backends import TensorCollection
from adaled.nn.activations import make_activation
from adaled.nn.initialization import apply_custom_initialization
from adaled.transformers.scaling import Scaling
from adaled.utils.dataclasses_ import \
        DataclassMixin, SPECIFIED_LATER, dataclass, field

import torch
import torch.nn.functional as F

from typing import Any, Optional, Sequence


@dataclass
class MLPConfig(DataclassMixin):
    dropout_keep_prob: float = 1.0
    activation: str = 'identity'
    activation_output: str = 'identity'
    layers_size: Sequence[int] = ()


@dataclass
class RichHiddenState:
    """Used to pass extra debug information through the hidden state of a
    recurrent neural network."""
    __slots__ = ('hidden', 'next_input', 'raw_output')
    hidden: Any
    next_input: Any
    raw_output: Any


@dataclass
class RNNConfig(DataclassMixin):
    """Specification of the RNN network.

    Hints:
        - If the autoencoder output is bounded to [-1, 1], it is reasonable to
          use tanh as the output activation and set residual to False.
    """
    input_size: int = SPECIFIED_LATER  # Required.
    output_size: int = 0  # If zero, set to `input_size`.
    rnn_hidden_size: int = 32
    rnn_num_layers: int = 1

    append_F: bool = True
    has_sigma2: bool = False
    cell_type: str = 'lstm'
    activation_output: str = 'identity'
    residual: bool = True
    detach_sigma2: bool = True

    scaling_F: Optional[Scaling] = None

    # The expected scale of dz (when residual == True). Used to scale the
    # output of the RNN. E.g. if dz ~ 0.01, use scale_dz ~ 0.01. Experimental.
    scale_dz: float = 1.0


class LinearResidualLayer(torch.nn.Module):
    """Computes linear(x) + x when input and output dimensions match and
    linear(x) when they don't.

    This optimization helps the training.
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.is_residual = input_dim == output_dim
        self.layer = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def __repr__(self):
        return f"{self.__class__.__name__}(layer={self.layer}, is_residual={self.is_residual})"

    def forward(self, x):
        if self.is_residual:
            return self.layer(x) + x
        else:
            return self.layer(x)


class PaddingLayer(torch.nn.Module):
    """Layer that performs torch.nn.functional.pad.

    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """
    def __init__(self, pad: Sequence[int], mode: str):
        """
        Arguments:
            pad: tuple of size 2, 4 or 6
                 (left, right, [top, bottom, [front, back]])
            mode: padding mode, see the link above
        """
        super().__init__()
        assert len(pad) % 2 == 0, pad
        self.pad = pad
        self.mode = mode

    def __repr__(self):
        return f"{self.__class__.__name__}(pad={self.pad}, mode={self.mode})"

    def forward(self, x):
        return F.pad(x, self.pad, self.mode)


# FIXME: Find a way to avoid this hack. Whether (x, F) is passed or just x
# should be controlled properly and not guessed.
class RemoveExternalForcingLayer(torch.nn.Module):
    """Converts TensorCollection(x=..., F=...) into just x."""
    LAYER_SHORT_NAME = "RemoveF"
    def forward(self, input):
        if isinstance(input, TensorCollection):
            # Hardcoding the TensorCollection(x=/z..., F=...) structure.
            if set(input.keys()) == set('xF'):
                return input['x']
        return input


class ViewLayer(torch.nn.Module):
    """Layer that resizes the input to the given shape."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def forward(self, x):
        return x.view(*self.shape)


class MLP(torch.nn.Module):
    def __init__(self, params: MLPConfig):
        super().__init__()

        layers = [RemoveExternalForcingLayer()]
        for i in range(len(params.layers_size) - 1):
            layers.append(LinearResidualLayer(params.layers_size[i],
                                              params.layers_size[i + 1],
                                              bias=True))
            if i < len(params.layers_size) - 2:
                layers.append(make_activation(params.activation))
                if params.dropout_keep_prob < 1.0:
                    layers.append(torch.nn.Dropout(p=1 - params.dropout_keep_prob))
            else:
                layers.append(make_activation(params.activation_output))

        self.layers = torch.nn.ModuleList(layers)
        apply_custom_initialization(self)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class SigmaLayer(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 # TODO: hidden_size should be configurable
                 hidden_size: int = 100,
                 activation: str = 'celu',
                 dropout_keep_prob: float = 1.0):
        super().__init__()

        layers = []
        layers.append(LinearResidualLayer(input_size, hidden_size, bias=True))
        layers.append(make_activation(activation))
        if dropout_keep_prob < 1.0:
            layers.append(torch.nn.Dropout(p=1 - dropout_keep_prob))
        layers.append(LinearResidualLayer(hidden_size, output_size, bias=True))
        layers.append(make_activation('softpluseps'))
        self.layers = torch.nn.ModuleList(layers)
        # apply_custom_initialization(self)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class RNN(torch.nn.Module):
    def __init__(self, params: RNNConfig):
        super().__init__()
        input_size  = params.input_size
        cell_type = params.cell_type
        output_size = params.output_size or input_size

        # Note: In principle, LSTMCell and GRUCell are sufficient here.
        if cell_type == 'lstm':
            self.rnn_cell = torch.nn.LSTM(
                    input_size=input_size, hidden_size=params.rnn_hidden_size,
                    num_layers=params.rnn_num_layers, batch_first=True)
        elif cell_type == 'gru':
            self.rnn_cell = torch.nn.GRU(
                    input_size=input_size, hidden_size=params.rnn_hidden_size,
                    num_layers=params.rnn_num_layers, batch_first=True)
        else:
            raise NotImplementedError(f"unrecognized cell type '{cell_type}'")

        self.output_layer = torch.nn.Linear(params.rnn_hidden_size, output_size)
        self.output_act = make_activation(params.activation_output)
        self.residual = params.residual
        self.append_F = params.append_F
        self.scaler_F = params.scaling_F.to_torch() if params.scaling_F else None
        if params.scale_dz != 1.0:
            if not self.residual:
                raise ValueError("scale_dz != 1.0 is only intended for residual == True at the moment")
            self.scaler_dz = Scaling(scale=params.scale_dz, shift=0.0).to_torch()
            self.scaler_dz_sqr = Scaling(scale=params.scale_dz ** 2, shift=0.0).to_torch()
        else:
            self.scaler_dz = self.scaler_dz_sqr = None
        self.detach_sigma2 = params.detach_sigma2

        if params.has_sigma2:
            self.sigma_layers = SigmaLayer(params.rnn_hidden_size, output_size)
        else:
            self.sigma_layers = None

        apply_custom_initialization(self)

    def forward(self, input, h=None):
        """Perform one RNN step.

        Arguments:
            input: TensorCollection(z=..., F=...),
                   where z is a tensor (batch_size, state_size)
                   and F a tensor (batch_size, external forcing shape...)
            h: hidden state

        Returns:
            (next z, hidden state)

        If external forcing is not provided to the network, `F` will be given
        as a 2D tensor of shape (batch_size, 0).
        """
        z = input['z']
        F = input['F']
        if F.shape[-1] > 0 and self.append_F:
            if self.scaler_F:
                F = self.scaler_F(F)
            # If individial Fs are scalars, convert them to 1D.
            if F.ndim == 1:
                F = F.reshape(F.shape + (1,))
            input_ = torch.cat([z, F], axis=-1)
        else:
            input_ = z

        # Add and remove the 2nd dimension (trajectory).
        out_rnn, h = self.rnn_cell(input_[:, None, :], h)
        out_rnn = out_rnn[:, 0, :]
        out = self.output_act(self.output_layer(out_rnn))
        if self.residual:
            if self.scaler_dz:
                out = self.scaler_dz(out)
            out = z + out

        if self.sigma_layers:
            if self.detach_sigma2:
                out_rnn = out_rnn.detach()
            sigma2 = self.sigma_layers(out_rnn)
            if self.scaler_dz:
                sigma2 = self.scaler_dz_sqr(sigma2)
            out = torch.stack([out, sigma2], dim=-1)
        return out, h
