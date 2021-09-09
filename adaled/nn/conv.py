from adaled.backends import TensorCollection
from adaled.nn.activations import make_activation
from adaled.nn.initialization import apply_custom_initialization
from adaled.nn.networks import PaddingLayer, RemoveExternalForcingLayer, ViewLayer
from adaled.nn.utility import describe_layers, get_layer_dimensions
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass, field

import numpy as np
import torch

from typing import Any, Dict, Optional, Sequence, Tuple, Union

_TupleOrInt = Union[Tuple[int, ...], int]

@dataclass
class _LayerClasses:
    Conv: torch.nn.Module
    ConvTranspose: torch.nn.Module
    BatchNorm: torch.nn.Module
    Dropout: torch.nn.Module
    MaxPool: torch.nn.Module
    AvgPool: torch.nn.Module


_nn = torch.nn
_layer_classes = [
    _LayerClasses(_nn.Conv1d, _nn.ConvTranspose1d, _nn.BatchNorm1d,
                  _nn.Dropout, _nn.MaxPool1d, _nn.AvgPool1d),
    _LayerClasses(_nn.Conv2d, _nn.ConvTranspose2d, _nn.BatchNorm2d,
                  _nn.Dropout2d, _nn.MaxPool2d, _nn.AvgPool2d),
    _LayerClasses(_nn.Conv3d, _nn.ConvTranspose3d, _nn.BatchNorm3d,
                  _nn.Dropout3d, _nn.MaxPool3d, _nn.AvgPool3d),
]
del _nn


@dataclass
class ConvMLPEncoderConfig(DataclassMixin):
    """Convolutional N-dim encoder with an MLP final layer."""
    input_shape: Tuple[int, ...] = SPECIFIED_LATER  # ([[z], y], x, channel)
    conv_layers_channels: Tuple[int, ...] = SPECIFIED_LATER
    conv_layers_kernel_sizes: _TupleOrInt = SPECIFIED_LATER
    conv_layers_strides: _TupleOrInt = SPECIFIED_LATER
    latent_state_dim: int = SPECIFIED_LATER

    pool_kernel_sizes: _TupleOrInt = SPECIFIED_LATER
    pool_kernel_strides: _TupleOrInt = SPECIFIED_LATER
    pool_type: str = 'avg'

    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # None means no padding.
    padding_mode: Optional[str] = 'circular'

    autoresize: bool = False

    dropout_keep_prob: float = 1.0
    activation: str = 'celu'
    activation_output: str = 'tanh'
    batch_norm: bool = True

    # Kwargs passed to ConvMLPDecoderConfig during make_decoder_config, with
    # higher priority than automatically computed kwargs, but with lower
    # priority than kwargs passed to make_decoder_config directly by the user.
    decoder_kwargs: Dict[str, Any] = field(dict)

    def fix(self):
        """Expand layer information specified as int into tuple of ints."""
        num_layers = len(self.conv_layers_channels)
        def _expand(a: _TupleOrInt):
            if isinstance(a, int):
                return (a,) * num_layers
            else:
                return a

        self.conv_layers_kernel_sizes = _expand(self.conv_layers_kernel_sizes)
        self.conv_layers_strides = _expand(self.conv_layers_strides)
        self.pool_kernel_sizes = _expand(self.pool_kernel_sizes)
        self.pool_kernel_strides = _expand(self.pool_kernel_strides)

    def validate(self, *args, **kwargs):
        super().validate(*args, **kwargs)
        self.assert_equal_length(r'conv_layers_.*', r'pool_kernel_.*')

    def total_scale_factor(self) -> int:
        """Return the product of all strides."""
        prod = 1
        for stride in self.conv_layers_strides:
            prod *= stride
        for stride in self.pool_kernel_strides:
            prod *= stride
        return prod

    def make_encoder(self) -> 'ConvMLPEncoder':
        """Create the encoder side of the autoencoder."""
        return ConvMLPEncoder(self)

    def make_autoencoder(self) -> 'adaled.transformers.autoencoders.AutoencoderTransformer':
        from adaled.transformers.autoencoders import AutoencoderTransformer
        encoder = self.make_encoder()
        decoder = encoder.make_decoder_config().make_decoder()
        return AutoencoderTransformer(encoder, decoder)



@dataclass
class ConvMLPDecoderConfig(DataclassMixin):
    """Convolutional N-dim decoder with an MLP first layer."""
    latent_state_dim: int
    first_layer_shape: Sequence[int]  # Shape without the number of channels.
    output_channels: int
    conv_layers_kernel_sizes: Sequence[int]
    conv_layers_channels: Sequence[int]
    conv_layers_strides: Sequence[int]

    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    upsampling_factors: Sequence[int]
    upsampling_mode: Optional[str] = None  # linear, bilinear or trilinear by default

    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # None means no padding.
    padding_mode: Optional[str] = 'circular'

    conv_transpose: bool = False
    dropout_keep_prob: float = 1.0
    activation: str = 'celu'
    activation_output: str = 'tanh'
    batch_norm: bool = True

    dim: int = 1

    def validate(self, *args, **kwargs):
        super().validate(*args, **kwargs)
        self.assert_equal_length('conv_layers_.*')

    def make_decoder(self) -> 'ConvMLPDecoder':
        """Create the decoder side of the autoencoder."""
        return ConvMLPDecoder(self)


def make_pooling(dim: int, pool_type: str, size: int, stride: int):
    classes = _layer_classes[dim - 1]
    if pool_type == 'max':
        return classes.MaxPool(size, stride=stride)
    elif pool_type == 'avg':
        return classes.AvgPool(size, stride=stride)
    else:
        raise ValueError("unrecognized pooling operation '{pool_type}'")


def _make_padding(
        kernel_size: int,
        is_conv_layer: bool,
        mode: str,
        dim: int) -> Tuple[Union[int, Tuple[int, int]], str, Sequence[torch.nn.Module]]:
    """Prepare padding and padding_mode arguments for Conv1d or an equivalent
    layer for pooling layers, since they do not have the same level of support
    for padding modes.

    Handled padding modes:
        None: no padding whatsoever
        circular: periodic boundary conditions, uses existing padding for
                  Conv1d and for odd kernel size, otherwise returns a padding
                  layer

    Returns:
        (padding, padding mode, list of padding layers)
        where padding is either an int or a tuple (left, right).
    """
    if mode is None:
        return (0, 'zeros', ())

    left = (kernel_size - 1) // 2
    right = kernel_size // 2
    if left == right and is_conv_layer:
        return (left, mode, ())  # Conv layer can handle it itself.
    else:
        if not is_conv_layer and mode == 'zeros':
            # Pooling and conv layers don't have the same inteface.
            mode = 'constant'
        layers = (PaddingLayer((left, right) * dim, mode),)
        return (0, 'zeros', layers)


class ConvMLPEncoder(torch.nn.Module):
    """1, 2 or 3-dim convolutional encoder."""
    def __init__(self, config: ConvMLPEncoderConfig):
        super().__init__()
        config.validate()
        if not (2 <= len(config.input_shape) <= 4):
            raise ValueError(f"expected 2, 3 or 4-dim input, "
                             f"got shape {config.input_shape} instead")
        dim = len(config.input_shape) - 1
        classes = _layer_classes[dim - 1]
        conv_layers_channels = [config.input_shape[0]] + config.conv_layers_channels
        layers = [RemoveExternalForcingLayer()]

        for i in range(len(config.conv_layers_kernel_sizes)):
            is_last_layer = i == len(config.conv_layers_kernel_sizes) - 1
            batch_norm = config.batch_norm and not is_last_layer

            padding, padding_mode, padding_layers = _make_padding(
                    config.conv_layers_kernel_sizes[i],
                    is_conv_layer=True, mode=config.padding_mode, dim=dim)
            layers.extend(padding_layers)
            layers.append(classes.Conv(
                    kernel_size=config.conv_layers_kernel_sizes[i],
                    in_channels=conv_layers_channels[i],
                    out_channels=conv_layers_channels[i + 1],
                    stride=config.conv_layers_strides[i],
                    padding=padding,
                    padding_mode=padding_mode,
                    # Batch norm cancels out the bias, so don't use it.
                    bias=not batch_norm))

            if batch_norm:
                layers.append(classes.BatchNorm(
                        conv_layers_channels[i + 1], affine=False))

            # TODO: Determine when exactly a padding layer is needed, when not.
            # padding, padding_mode, padding_layers = _make_padding(
            #         config.pool_kernel_sizes[i], is_conv_layer=False,
            #         mode=config.padding_mode, dim=dim)
            # layers.extend(padding_layers)
            if config.pool_kernel_sizes[i] > 1:
                layers.append(make_pooling(
                        dim,
                        config.pool_type,
                        config.pool_kernel_sizes[i],
                        config.pool_kernel_strides[i]))
            layers.append(make_activation(config.activation))

            if not is_last_layer:
                if config.dropout_keep_prob < 1.0:
                    layers.append(classes.Dropout(p=1 - config.dropout_keep_prob))

        # Add the MLP layer.
        shapes = get_layer_dimensions(layers, config.input_shape)
        layers.append(torch.nn.Flatten(start_dim=-1 - dim, end_dim=-1))
        layers.append(torch.nn.Linear(
                np.prod(shapes[-1]),
                config.latent_state_dim,
                bias=True))
        layers.append(make_activation(config.activation_output))

        self.config = config
        self.ndim = dim
        self.layers = torch.nn.ModuleList(layers)
        self.last_conv_layer_shape = shapes[-1]

        apply_custom_initialization(self)

    def describe(self):
        config = self.config
        # from torchinfo import summary
        # summary(self, input_data=torch.randn(1, *config.input_shape))
        describe_layers(self, self.layers, config.input_shape)

    def __iter__(self):
        return iter(self.layers)

    def make_decoder_config(self, **kwargs):
        """Create decoder config for a CNN with the reversed structure.

        Any `kwargs` is forwarded to the decoder config and overrides the
        default values.
        """
        config = self.config
        upsampling_factors = \
                [x * y for x, y in zip(config.conv_layers_strides[::-1],
                                       config.pool_kernel_strides[::-1])]
        dim = len(config.input_shape) - 1
        assert 1 <= dim <= 3
        def_kwargs = {
            'latent_state_dim': config.latent_state_dim,
            'first_layer_shape': self.last_conv_layer_shape[1:],
            'output_channels': config.input_shape[0],
            'conv_layers_kernel_sizes': config.conv_layers_kernel_sizes[::-1],
            'conv_layers_channels': config.conv_layers_channels[::-1],
            'padding_mode': config.padding_mode,
            'dropout_keep_prob': config.dropout_keep_prob,
            'activation': config.activation,
            'activation_output': config.activation_output,
            'batch_norm': config.batch_norm,
        }
        extra_kwargs = {**config.decoder_kwargs, **kwargs}
        if extra_kwargs.get('conv_transpose', False):
            # For ConvTranspose we move the scaling to the conv layer.
            def_kwargs.update({
                'conv_layers_strides': upsampling_factors,
                'upsampling_factors': [1] * len(upsampling_factors),
            })
        else:
            def_kwargs.update({
                'conv_layers_strides': [1] * len(config.conv_layers_strides),
                'upsampling_factors': upsampling_factors,
                'upsampling_mode': ('linear', 'bilinear', 'trilinear')[dim - 1],
            })
        def_kwargs.update(extra_kwargs)
        # FIXME: If too large strides are used, it may happen that decoder
        # produces an image of a different resolution than the encoder's input.
        return ConvMLPDecoderConfig(**def_kwargs)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class ConvMLPDecoder(torch.nn.Module):
    def __init__(self, config: ConvMLPDecoderConfig):
        super().__init__()
        config.validate()
        dim = len(config.first_layer_shape)
        if not (1 <= dim <= 3):
            raise ValueError(f"expected 1D, 2D or 3D, got {config.first_layer_shape}")

        self.config = config
        self.layers = torch.nn.ModuleList(self.create_layers(config))
        apply_custom_initialization(self)

    def create_layers(self, config: ConvMLPDecoderConfig):
        layers = []
        layers.extend(self.create_expansion_layers(config))
        layers.extend(self.create_conv_layers(config))
        if config.activation_output and config.activation_output != 'none':
            layers.append(make_activation(config.activation_output))
        return layers

    def create_expansion_layers(self, config: ConvMLPDecoderConfig):
        """Create layers that expand the latent state to the first image state."""
        first_num_channels = config.conv_layers_channels[0]
        layer1 = torch.nn.Linear(
                config.latent_state_dim,
                first_num_channels * np.prod(config.first_layer_shape),
                bias=True)
        layer2 = ViewLayer((-1, first_num_channels, *config.first_layer_shape))
        return [layer1, layer2]

    def create_conv_layers(self, config: ConvMLPDecoderConfig):
        dim = len(config.first_layer_shape)
        conv_layers_channels = config.conv_layers_channels + [config.output_channels]
        classes = _layer_classes[dim - 1]

        layers = []
        for i in range(len(config.conv_layers_kernel_sizes)):
            batch_norm = config.batch_norm \
                    and i < len(config.conv_layers_kernel_sizes) - 1

            if config.upsampling_factors[i] > 1:
                layers.append(torch.nn.Upsample(
                        scale_factor=config.upsampling_factors[i],
                        mode=config.upsampling_mode, align_corners=False))

            if config.conv_transpose:
                conv_cls = classes.ConvTranspose
            else:
                conv_cls = classes.Conv

            padding, padding_mode, padding_layers, conv_kwargs = self.make_padding(i)
            layers.extend(padding_layers)
            layers.append(conv_cls(
                    kernel_size=config.conv_layers_kernel_sizes[i],
                    in_channels=conv_layers_channels[i],
                    out_channels=conv_layers_channels[i + 1],
                    stride=config.conv_layers_strides[i],
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=not batch_norm,
                    **conv_kwargs))

            if batch_norm:
                layers.append(classes.BatchNorm(
                        conv_layers_channels[i + 1], affine=False))

            if i < len(config.conv_layers_kernel_sizes) - 1:
                layers.append(make_activation(config.activation))
                if config.dropout_keep_prob < 1.0:
                    layers.append(classes.Dropout(p=1 - config.dropout_keep_prob))
        return layers

    def make_padding(self, layer: int) \
            -> Tuple[Union[int, Tuple[int, int]], str,
                     Sequence[torch.nn.Module], Dict[str, Any]]:
        """Returns padding layers (if Conv layer does not support this
        padding), or Conv-compatible padding size and padding mode.

        Returns:
            padding: int or pair of ints
            padding_mode: str
            padding_layers: list
            conv_kwargs: dict, passed to the convolutional layer constructor
        """
        config = self.config
        if config.conv_transpose:
            # Not sure about the padding here.
            return (config.conv_layers_strides[layer] - 1, 'zeros', (), {})
        dim = len(config.first_layer_shape)
        out = _make_padding(config.conv_layers_kernel_sizes[layer], dim=dim,
                             is_conv_layer=True, mode=config.padding_mode)
        return (*out, {})

    def describe(self):
        input_shape = (self.config.latent_state_dim,)
        # from torchinfo import summary
        # summary(self, input_data=torch.randn(1, *input_shape))
        describe_layers(self, self.layers, input_shape)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
