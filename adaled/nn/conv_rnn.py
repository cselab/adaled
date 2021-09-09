# Created by: Vlachas Pantelis, Ivica Kicic, CSE-lab, ETH Zurich

from adaled.nn.activations import make_activation
from adaled.utils.dataclasses_ import \
        DataclassMixin, SPECIFIED_LATER, dataclass, field

import torch
from torch.nn import Parameter

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

__all__ = ['ConvRNNConfig', 'ConvRNN']

def _layer_arg(x: Union[Any, List[Any]], i: int):
    """Return x[i] if x is a list, otherwise x.

    Used to expand the configuration parameters for ConvRNN layers.
    """
    return x[i] if isinstance(x, list) else x


@dataclass
class ConvRNNConfig(DataclassMixin):
    input_channels: int = SPECIFIED_LATER  # Required.
    layer_channels: List[int] = SPECIFIED_LATER

    kernel_sizes: Union[List[int], int] = 5
    dilations: Union[List[int], int] = 1
    cell_type: str = 'lstm'

    ndim: int = SPECIFIED_LATER
    append_F: bool = True
    residual: bool = True

    cell_kwargs: Dict[str, Any] = field(dict)

    def make(self):
        return ConvRNN(self)

    def total_kernel_range(self):
        """Return how far the information travels from one input element.

        >>> Config(input_channels=[2], kernel_sizes=[5]).total_kernel_range()
        2
        >>> Config(input_channels=[2], kernel_sizes=[7]).total_kernel_range()
        3
        >>> Config(input_channels=[2, 2], kernel_sizes=[5, 5]).total_kernel_range()
        4
        """
        size = 1
        for i in range(len(self.layer_channels)):
            k = _layer_arg(self.kernel_sizes, i)
            if k % 2 == 0:
                raise NotImplementedError("even kernel size not supported")
            d = _layer_arg(self.dilations, i)
            size = size - 1 + k + (k - 1) * (d - 1)
        return size // 2

    def _get_cell_kwargs(self, i: int) -> Dict[str, Any]:
        kwargs = {k: _layer_arg(x, i) for k, v in self.cell_kwargs.items()}
        kwargs['ndim'] = self.ndim
        kwargs['dilation'] = _layer_arg(self.dilations, i)
        kwargs['kernel_size'] = _layer_arg(self.kernel_sizes, i)
        kwargs['cell_type'] = self.cell_type
        return kwargs


class ConvRNNCell(torch.nn.Module):
    """1D or 2D convolutional RNN cell."""
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 *,
                 activation: str = 'tanh',
                 cell_type: str = 'lstm',
                 dilation: int = 1,
                 torch_dtype: Union[type, Callable] = torch.tensor,
                 ndim: Optional[int] = None,
                 shape: Optional[Sequence[int]] = None):
        super().__init__()

        if shape is None and cell_type in ['lstm_2', 'lstm_3']:
            raise TypeError(f"`shape` must be provided for cell_type={cell_type}")
        if ndim is None and shape is None:
            raise TypeError("specify one of `ndim` and `shape`")
        if ndim is not None and shape is not None and ndim != len(shape):
            raise ValueError(f"inconsistent dimensionality ndim={ndim} shape={shape}")
        elif ndim is None:
            ndim = len(shape)

        if ndim == 1:
            Conv = torch.nn.Conv1d
        elif ndim == 2:
            Conv = torch.nn.Conv2d
        else:
            raise ValueError(f"expected 1D or 2D, got ndim={ndim} shape={shape}")

        self.hidden_channels = hidden_channels
        self.rnn_cell_type = cell_type
        self.activation = make_activation(activation)

        if kernel_size % 2 == 0:
            raise NotImplementedError(
                    "even kernel_size not supported (padding not implemented)")
        eff_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = (eff_kernel_size - 1) // 2
        kw = dict(kernel_size=kernel_size, padding=padding, dilation=dilation)
        if cell_type == 'lstm':
            self.Wxi = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxf = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxc = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxo = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Whi = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whf = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whc = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Who = Conv(hidden_channels, hidden_channels, bias=True, **kw)

        elif cell_type == 'lstm_2':
            self.Wxi = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxf = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxc = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxo = Conv(input_channels, hidden_channels, bias=True, **kw)

            self.Whi = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whf = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whc = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Who = Conv(hidden_channels, hidden_channels, bias=True, **kw)

            self.Bci = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bcf = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bco = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bcc = Parameter(torch_dtype(1, hidden_channels, *shape))

        elif cell_type == 'lstm_3':
            self.Wxi = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxf = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxc = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Wxo = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Whi = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whf = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Whc = Conv(hidden_channels, hidden_channels, bias=True, **kw)
            self.Who = Conv(hidden_channels, hidden_channels, bias=True, **kw)

            self.Wci = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Wcf = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Wco = Parameter(torch_dtype(1, hidden_channels, *shape))

            self.Bci = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bcf = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bco = Parameter(torch_dtype(1, hidden_channels, *shape))
            self.Bcc = Parameter(torch_dtype(1, hidden_channels, *shape))

        elif cell_type == 'gru':
            self.Wiz = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Whz = Conv(hidden_channels, hidden_channels, bias=False, **kw)
            self.Wir = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Whr = Conv(hidden_channels, hidden_channels, bias=False, **kw)
            self.Wih = Conv(input_channels, hidden_channels, bias=True, **kw)
            self.Whh = Conv(hidden_channels, hidden_channels, bias=False, **kw)

        else:
            raise ValueError("unknown cell_type: " + cell_type)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden_state(x)

        # gi == input gate
        # gf == forget gate
        # go == output gate
        # gc == cell gate
        if self.rnn_cell_type == 'lstm':
            h_prev = hidden[0]
            c_prev = hidden[1]

            gi = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
            gf = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
            go = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
            gc = self.activation(self.Wxc(x) + self.Whc(h_prev))

            c_next = (gf * c_prev) + (gi * gc)
            h_next = go * self.activation(c_next)

            next_hidden = h_next, c_next
            output = h_next

        elif self.rnn_cell_type == 'lstm_2':
            h_prev = hidden[0]
            c_prev = hidden[1]

            gi = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev) + self.Bci)
            gf = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev) + self.Bcf)
            gc = self.activation(self.Wxc(x) + self.Whc(h_prev) + self.Bcc)
            go = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.Bco)

            c_next = (gf * c_prev) + (gi * gc)
            h_next = go * self.activation(c_next)

            next_hidden = h_next, c_next
            output = h_next

        elif self.rnn_cell_type == 'lstm_3':
            h_prev = hidden[0]
            c_prev = hidden[1]

            gi = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev) + c_prev * self.Wci + self.Bci)
            gf = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev) + c_prev * self.Wcf + self.Bcf)
            gc = self.activation(self.Wxc(x) + self.Whc(h_prev) + self.Bcc)

            c_next = (gf * c_prev) + (gi * gc)
            go = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + c_next * self.Wco + self.Bco)
            h_next = go * self.activation(c_next)

            next_hidden = h_next, c_next
            output = h_next

        elif self.rnn_cell_type == 'gru':
            h_prev = hidden
            z = torch.sigmoid(self.Wiz(x) + self.Whz(h_prev))
            r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))
            h_next = self.activation(self.Wih(x) + self.Whh(r * h_prev))
            output = (1. - z) * h_prev + z * h_next
            next_hidden = output
        return output, next_hidden

    def init_hidden_state(self, input_batch):
        shape = input_batch.shape
        batch_size = shape[0]
        spatial_size = shape[2:]
        state_size = [batch_size, self.hidden_channels] + list(spatial_size)
        if self.rnn_cell_type == 'gru':
            hidden = torch.zeros(state_size)
        else:
            hidden = (torch.zeros(state_size), torch.zeros(state_size))
        return hidden


class ConvRNN(torch.nn.Module):
    def __init__(self, config: ConvRNNConfig):
        super().__init__()
        config.validate()

        layer_channels = [config.input_channels, *config.layer_channels]
        cells = [
            ConvRNNCell(input_channels=layer_channels[i],
                        hidden_channels=layer_channels[i + 1],
                        **config._get_cell_kwargs(i))
            for i in range(len(layer_channels) - 1)
        ]
        self.cells = torch.nn.ModuleList(cells)
        self.append_F = config.append_F
        self.residual = config.residual

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = [None] * len(self.cells)

        z = input['z']
        F = input['F']
        if F.shape[-1] > 0 and self.append_F:
            raise NotImplementedError("append_F=True not supported")
            # If individial Fs are scalars, convert them to 1D.
            if F.ndim == 1:
                F = F.reshape(F.shape + (1,))
            # F = torch.broadcast_to(F, z.shape[:1] + z.shape[2:])
            out = torch.cat([z, F], axis=-1)
        else:
            out = z

        new_hidden = []
        for cell, h in zip(self.cells, hidden):
            out, h = cell(out, h)
            new_hidden.append(h)

        if self.residual:
            out = z + out

        return out, new_hidden
