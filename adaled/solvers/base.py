from adaled.backends import TensorCollection
import adaled.backends as backends

import numpy as np

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union


_Array = Union[np.ndarray, 'torch.Tensor']

class MacroSolver:
    """Propagator in the latent space.

    Computes the next state of a system and the uncertainty of the
    prediction."""
    def __init__(self):
        self.metadata: Dict[str, Any] = {'version': np.int64(0)}

    def state_dict(self) -> Dict[str, Any]:
        return {'metadata': self.metadata}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.metadata = state['metadata']

    def train(self, mode: bool):
        """Set the model to training or evaluation mode."""
        # TODO: Base everything on torch.nn.Module to avoid having these issues
        # with train / state_dict / load_state_dict etc?
        pass

    def advance(
            self,
            batch: TensorCollection,
            hidden=None, *,
            compute_uncertainty: bool = False,
            teacher_forcing: bool = False) \
                    -> Tuple[_Array, Optional[_Array], Optional[_Array]]:
        """Advance the `batch` for `steps` steps and compute the result uncertainty.

        Arguments:
            batch: `TensorCollection(z=..., F=...)`, a batch of states and external forcings
            hidden: (optional) data of unspecified type used to pass hidden state
            compute_uncertainty: (bool) whether to output the uncertainty or to
                    return `None`, defaults to `False`
            teacher_forcing: (bool) some macro solvers may have internal
                             mechanisms for propagating results from previous
                             steps (see ensembles), this tells them to enforce
                             the usage of the input `batch`

        Note that, as opposed to `torch.nn.LSTM`, here the input tensor does
        not contain the trajectory (temporal) dimension!

        Returns a tuple of:
            output: tensor-like object of shape `(batch_size,) + z.shape`
            hs: hidden state of unspecified type
            uncertainty: tensor of shape `(batch_size,)` or `None`
        """
        raise NotImplementedError()

    def advance_multiple(
            self,
            z_batch: 'arraylike',
            hidden: Optional[Any] = None,
            *,
            extra_steps: int,
            F_batches: Optional[Iterable[Any]] = None,
            compute_uncertainty: bool = False) -> 'arraylike':
        """
        Apply teacher forcing for the given time steps (`z_batch`) and continue
        advancing the system for `extra_steps` steps.

        Note: The argument `z_batch` is different than the argument `batch` in
        `advance`. Here, the `z_batch` does not include the external forcing,
        while `batch` in `advance` includes both `z` and `F`. The external
        forcing is instead provided separately as an iterable with at least
        `steps` number of elements.

        Arguments;
            z_batch: (collection-like) batch of latent trajectories,
                     shape (batch size, number of teacher forcing steps, ...)
            hidden: (any) hidden state
            extra_steps: (int) number of extra steps to advance
            F_batches: (optional) a list of external forcings with length at 
                       least equal to the number of teacher forcing steps +
                       `extra_steps`; if not provided, external forcing of
                       shape (0,) will be used
            compute_uncertainty: whether to compute and output uncertainty or
                    to return `None`

        Returns:
            trajectories: (batch_size, 1 + teacher forcing steps + extra_steps, ...)
            hidden: hidden states
            uncertainties: (batch_size, teacher forcing steps + extra_steps, ...)

        The output trajectory always includes the initial `z`.
        """
        available_z_steps = len(z_batch[0])

        if F_batches is None:
            _F_batch = backends.get_backend(z_batch).zeros((len(z_batch), 0))
            F_batches = [_F_batch] * (available_z_steps + extra_steps)

        out = backends.extended_emptylike(
                z_batch[:, 0], 1 + available_z_steps + extra_steps, axis=1)
        out[:, 0] = z_batch[:, 0]
        out_uncertainty = None

        for i, F in zip(range(available_z_steps + extra_steps), F_batches):
            if i < available_z_steps:
                z = z_batch[:, i]
            assert len(z) == len(F), (z.shape, F.shape)
            batch = TensorCollection(z=z, F=F)
            z, hidden, *uncertainty = self.advance(
                    batch, hidden, compute_uncertainty=compute_uncertainty,
                    teacher_forcing=(i < available_z_steps))
            out[:, 1 + i] = z
            if compute_uncertainty:
                uncertainty = uncertainty[0]
                if i == 0:
                    out_uncertainty = backends.extended_emptylike(
                            uncertainty, available_z_steps + extra_steps, axis=1)
                out_uncertainty[:, i] = uncertainty

        return (out, hidden, out_uncertainty)


class MicroSolver:
    """Propagator in the real space. Owns the system state."""

    def state_dict(self) -> dict:
        """Serialization of the whole micro solver.
        By default returns `{'state': self.get_state()}`."""
        return {'state': self.get_state()}

    def load_state_dict(self, state: dict) -> None:
        """Unserialization of the micro solver. By default assumes `state` is a
        dictionary `{'state': ...}`."""
        self.update_state(state['state'], skip_steps=0)

    def advance(self, F_batch: Sequence[Any]) -> 'arraylike':
        """Advance the system by one time steps.

        Arguments:
            F_batch: a batch of F, with one element per simulation

        Returns the batch of new state.
        """
        raise NotImplementedError()

    def get_state(self) -> 'arraylike':
        """Return the current state."""
        raise NotImplementedError()

    def update_state(self, new_state: 'arraylike', skip_steps: int = 1):
        """Update the state and skip the given number of discrete steps."""
        raise NotImplementedError()

    def close(self):
        """Close the micro solver. No-op by default. Useful for multirank micro
        solvers."""
        pass
