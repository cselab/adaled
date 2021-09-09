from adaled.backends import TensorCollection, get_backend
from adaled.nn.trainers import AutoencoderTrainer, TrainingConfig
from adaled.transformers.scaling import Scaling
from adaled.transformers.base import AutoencoderModule, Transformer
from adaled.nn.tracer import get_tracer
import adaled

import torch

from collections import OrderedDict
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

RecDict = Dict  # TODO: Recursive dictionary.

def _cumulative_sum(xs):
    out = [0]
    for x in xs:
        out.append(out[-1] + x)
    return out


class AutoencoderTransformer(Transformer):
    """Wraps an autoencoder encoder and decoder into a Transformer.

    Ensures that the encoder and decoders get the input data in the correct
    format (backend).

    Arguments:
        encoder: encoder part of the autoencoder
        decoder: decoder part of the autoencoder
        backend: backend of the encoder and decoder (defaults to torch):
    """

    def __init__(self, encoder: Callable, decoder: Callable,
                 backend: Optional[Union[str, adaled.Backend]] = 'torch'):
        if isinstance(backend, str):
            if backend == 'torch':
                backend = get_backend(torch.arange(5))
            elif backend == 'auto':
                backend = get_backend(next(encoder.parameters()))
            else:
                raise ValueError(backend)

        self.backend = backend
        self.model = AutoencoderModule(encoder, decoder)

    def transform(self, x):
        """Apply :attr:`encoder` on :attr:`x`."""
        if self.backend:
            x = self.backend.cast_from(x)
        return get_tracer().evaluate(self.model.encoder, x)

    def inverse_transform(self, z):
        """Apply :attr:`decoder` on :attr:`z`."""
        if self.backend:
            z = self.backend.cast_from(z)
        return get_tracer().evaluate(self.model.decoder, z)

    def make_trainer(self,
                     config: TrainingConfig,
                     cls: type = AutoencoderTrainer,
                     **kwargs):
        """Make a trainer that trains its model to output the same values as
        the input."""
        return cls(self.model, **config.make(self.model.parameters()), **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class _CompoundEncoder(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
        self.models = torch.nn.ModuleList(
                [trans.model.encoder for trans in mapping.values()])

    def forward(self, x):
        # FIXME: Do not hack this. AdaLED Should pass only the 'x' part to the autoencoders.
        if len(x.keys()) == 2 and 'x' in x and 'F' in x:
            x = x['x']
        if x.keys() != self.mapping.keys():
            self_keys = ', '.join(self.mapping.keys())
            x_keys = ', '.join(x.keys())
            raise ValueError(f"expected keys ({self_keys}), got ({x_keys})")
        zs = []
        tracer = get_tracer()
        for key, trans in self.mapping.items():
            with tracer(key):
                zs.append(trans.transform(x[key]))

        # First element defines the backend.
        backend = get_backend(zs[0])
        return backend.cat([backend.cast_from(z) for z in zs], axis=-1)


class _CompoundDecoder(torch.nn.Module):
    def __init__(self, mapping, latent_sizes: Sequence[int]):
        super().__init__()
        self.mapping = mapping
        self._offsets = _cumulative_sum(latent_sizes)
        self.models = torch.nn.ModuleList(
                [trans.model.decoder for trans in mapping.values()])

    def forward(self, z):
        offsets = self._offsets
        if z.ndim != 2:
            raise TypeError(f"expected 2D tensor, got shape {z.shape}")
        tracer = get_tracer()
        out = {}
        for i, (key, trans) in enumerate(self.mapping.items()):
            with tracer(key):
                out[key] = trans.inverse_transform(z[:, offsets[i] : offsets[i + 1]])
        return TensorCollection(out)


class CompoundAutoencoder(AutoencoderTransformer):
    """Distributes different components of a :class:`TensorCollection` state to
    different autoencoders and concatenates their results.

    Arguments:
        mapping: hierarchical mapping from keys to transformers and their
                 latent state sizes

    Example:
        >>> a = TransformerA()  # E.g. encodes to latent state of size 3.
        >>> b = TransformerB()  # E.g. encodes to latent state of size 5.
        >>> ab = CompoundAutoencoder({'a': (a, 3), 'b': (b, 5)})
        >>> x = TensorCollection(a=..., b=...)
        >>> z = ab(x)
        >>> (z[:, 0:3] == a(x['a'])).all()
        True
        >>> (z[:, 3:8] == b(x['b'])).all()
        True
    """

    def __init__(self, mapping: RecDict[str, Tuple[Transformer, int]], **kwargs):
        latent_sizes = []
        ae_mapping: Dict[str, Transformer] = OrderedDict()
        for key, value in mapping.items():
            if isinstance(value, (TensorCollection, dict)):
                transformer = self.__class__(value, **kwargs)
                latent_size = transformer.model.decoder._offsets[-1]
            else:
                transformer, latent_size = value
            ae_mapping[key] = transformer
            latent_sizes.append(latent_size)

        encoder = _CompoundEncoder(ae_mapping)
        decoder = _CompoundDecoder(ae_mapping, latent_sizes)
        super().__init__(encoder, decoder, **kwargs)
        self.ae_mapping = ae_mapping

    def __getitem__(self, key: str) -> Union['CompoundAutoencoder', Transformer]:
        """Return the autoencoder corresponding to the key :attr:`key`."""
        return self.ae_mapping[key]

    def slice_latent(self, data: 'arraylike', key: str) -> 'arraylike':
        """Return the part of a batch of latent states (output of
        ``.transform()``) that corresponds to the given :attr:`key`.

        Slicing is done with respect to the axis 1, the batch dimension.
        """
        offsets = self.model.decoder._offsets
        i = list(self.ae_mapping.keys()).index(key)
        return data[:, offsets[i] : offsets[i + 1]]

    def partial_inverse_transform(
            self, data: 'collectionlike', key: str) -> 'collectionlike':
        """Inverse-transform part of the data corresponding to :attr:`key`.

        Only the decoder correspoding to :attr:`key` is evaluated.
        """
        assert len(data[0]) == self.model.decoder._offsets[-1], \
               (len(data[0]), self.model.decoder._offsets,
                       self.model.decoder.mapping.keys(), data.shape)
        data = self.slice_latent(data, key)
        return self.ae_mapping[key].inverse_transform(data)


class ScaledAutoencoderTransformer(AutoencoderTransformer):
    """Autoencoder that scales and shifts the input.

    Args:
        encoder: encoder part of the autoencoder
        decoder: decoder part of the autoencoder
        scaling: scaling and shifting to apply
    """
    def __init__(self,
                 encoder: Callable,
                 decoder: Callable,
                 scaling: Scaling,
                 inv_scaling: Optional[Scaling] = None,
                 **kwargs):
        if inv_scaling is None:
            inv_scaling = scaling.inversed()
        encoder = torch.nn.Sequential(scaling.to_torch(), encoder)
        decoder = torch.nn.Sequential(decoder, inv_scaling.to_torch())
        super().__init__(encoder, decoder, **kwargs)
