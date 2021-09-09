from adaled.backends import TensorCollection
import adaled

import torch

from typing import Any, List
import copy

__all__ = [
    'Transformer', 'IdentityTransformer', 'StackedTransformer',
]

# TODO: Rename transformers to Autoencoders and rename
#       transform/inverse_transform to encode/decode?
# TODO: Replace Transfomer with AutoencoderModule everywhere?

class AutoencoderModule(torch.nn.Module):
    """Simple autoencoder network composed of an encoder and decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


class Transformer:
    """Base class for transformers.

    Transforms data from one representation to another and back.
    """
    __slots__ = ()

    def __call__(self, x):
        """Alias of self.transform."""
        return self.transform(x)

    def transform(self, x: Any) -> Any:
        """Transform a batch of real (micro) states :attr:`x` to the latent
        (macro) state :attr:`z`."""
        raise NotImplementedError()

    def transform_trajectory_batch(
            self, batch: TensorCollection, states_per_batch: int):
        """Flatten trajectories to states, transform, and reshape back.

        Args:
            batch: a batch of trajectories
            states_per_batch: processing batch size, number of trajectories to
                transform in one go

        Note that the term `batch` in :attr:`batch` denotes that the
        trajectories are given as a single :class:`TensorCollection` (as
        opposed to a list of collections), whereas the `batch` in
        :attr:`states_per_batch` is about the number of simulatenously
        processed trajectories.
        """
        x = batch['x']
        old_shape = (len(x), len(x[0]))
        total_states = old_shape[0] * old_shape[1]

        # Merging will cause copying if data is not contiguous. This is not a
        # big problem unless we work with large states, but there there is only
        # 1 trajectory anyway.
        x = adaled.merge_axes_01(x)
        if states_per_batch == 0 or states_per_batch >= total_states:
            # Transfer everything in one go.
            z = self.transform(x)
        else:
            # Transfer in chunks of given size.
            z = [self.transform(x[i : i + states_per_batch])
                 for i in range(0, len(x), states_per_batch)]
            z = adaled.cmap(lambda *a: adaled.get_backend(a[0]).cat(a), *z)
        z = adaled.split_first_axis(z, old_shape)
        return TensorCollection(z=z, F=batch['F'])

    def inverse_transform(self, z):
        """Inversely transform a batch of latent states :attr:`z` to real
        states :attr:`x`."""
        raise NotImplementedError()

    def to_latent_and_back(self, x):
        """Transform a real state :attr:`x` to latent and transform back."""
        return self.inverse_transform(self.transform(x))

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state):
        raise NotImplementedError()


class IdentityTransformer(Transformer):
    """Identity transformation, leaves the input unchanged."""
    __slots__ = ('model',)

    def __init__(self):
        encoder = decoder = torch.nn.Identity()
        self.model = AutoencoderModule(encoder, decoder)

    def transform(self, x):
        """Returns a copy of :attr:`x`.

        A copy is performed in order to avoid unexpected bugs.
        """
        # torch.Tensors don't like deepcopy, call clone manually.
        if hasattr(x, 'clone'):
            return x.clone()
        elif hasattr(x, 'copy'):
            return x.copy()
        else:
            return copy.deepcopy(x)

    def inverse_transform(self, z):
        """Return a copy of :attr:`z`.

        A copy is performed in order to avoid unexpected bugs."""
        return self.transform(z)  # Copy.

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert len(state) == 0


class StackedTransformer(Transformer):
    """Stacks multiple transformers.

    Arguments:
        transformers (list of transformers):
            list of child transformers, in the order of execution

    Examples:
        >>> a = TransformerA()
        >>> b = TransformerB()
        >>> ab = StackedTransformer([a, b])
        >>> # Equivalent to b.transform(a.transform(x)):
        >>> z = ab.transform(x)
        >>> # Equivalent to a.inverse_transform(b.inverse_transform(x)):
        >>> x = ab.inverse_transform(z)
    """

    def __init__(self, transformers: List[Transformer]):
        self.transformers = transformers

    def transform(self, x):
        for transformer in self.transformers:
            x = transformer.transform(x)
        return x  # Final x is z.

    def inverse_transform(self, z):
        for transformer in self.transformers[::-1]:
            z = transformer.inverse_transform(z)
        return z  # Final z is x.

    def state_dict(self):
        return {i: t.state_dict() for i, t in enumerate(self.transformers)}

    def load_state_dict(self, state_dict):
        assert len(self.transformers) == len(state_dict)
        for i, state in state_dict.items():
            self.transformers[i].load_state_dict(state)
