from adaled.backends import TensorCollection, cmap, get_backend
from adaled.backends.backend_torch import TorchBackend
from adaled.nn.networks import RichHiddenState
from adaled.nn.trainers import NNEnsembleTrainer, RNNTrainer, \
        RNNTrainerWithAdversarialLoss,TrainingConfig
from adaled.nn.loss import ProbabilisticLosses
from adaled.solvers.base import MacroSolver
import torch

from typing import Optional, Sequence

# TODO: Should this file be in adaled/solvers/ or in adaled/nn/, since the
# ensemble is not restricted to RNNs?

class Ensemble(MacroSolver):
    """Macro solver that computes uncertainty using an ensemble of propagators.

    This is an abstract class. See `process_ensemble_output`.

    Arguments:
        propagators: list of individual propagators
    """
    def __init__(self, propagators: Sequence):
        super().__init__()
        self.propagators = propagators

    def state_dict(self):
        out = super().state_dict()
        for i, p in enumerate(self.propagators):
            out[i] = p.state_dict()
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for i, p in enumerate(self.propagators):
            self.propagators[i].load_state_dict(state_dict[i])

    def train(self, mode: bool):
        for propagator in self.propagators:
            propagator.train(mode)

    def __iter__(self):
        return iter(self.propagators)

    def __len__(self):
        return len(self.propagators)

    def advance(self, batch, hidden: Optional[RichHiddenState] = None, *,
                compute_uncertainty: bool = False,
                teacher_forcing: bool = False):
        """Advance each of the networks and return the mean and optionally the
        uncertainty.

        Arguments:
            batch: (any) the batch of states to propagate
            hidden: (RichHiddenState, optional) hidden state, contains previous
                    output of individual propagators, passed instead of the
                    `batch` unless `teacher_forcing` is set
            teacher_forcing: enforce propagation of the `batch` instead of
                             outputs stored in `hidden`

        Return values:
            mean: the mean computed by `process_ensemble_output`
            hs: (RichHiddenState) the internal hidden state and the raw output
                of the ensemble in shape (num simulations, ensemble size, ...),
                useful for debugging and extra analysis
            uncertainties: (optional) uncertainties

        """
        ensemble_size = len(self.propagators)
        def preprocess(x):
            """Convert to torch and broadcast along the ensemble."""
            x = torch.as_tensor(x)
            return x.broadcast_to((ensemble_size,) + x.shape)

        batch = cmap(preprocess, batch)
        if hidden is None:
            hidden = [None] * ensemble_size
        else:
            assert isinstance(hidden, RichHiddenState)
            if not teacher_forcing and hidden.next_input is not None:
                batch = TensorCollection(z=hidden.next_input, F=batch['F'])
            hidden = hidden.hidden

        ensemble_output = None
        hs = [None] * ensemble_size
        for i, input, propagator, h in \
                zip(range(ensemble_size), batch, self.propagators, hidden):
            out, h = propagator(input, h)
            if i == 0:
                # Allocate buffers for the output of the whole ensemble, once
                # the output shape is known.
                ensemble_output = cmap(
                        lambda x: get_backend(x).empty((ensemble_size,) + x.shape), out)
            ensemble_output[i] = out
            hs[i] = h

        # Compute the mean and optionally uncertainty along the batch dimension.
        mean, next_input, *optional_uncertainty = \
                self.process_ensemble_output(
                    ensemble_output, compute_uncertainty=compute_uncertainty)

        # Store output in the hidden state, for potential recording and
        # analysis. Put the most general axis num_simulations at start, to hide
        # the internal details about the ensemble for codes that don't care
        # about it.
        ensemble_output = cmap(lambda x: get_backend(x).moveaxis(x, 0, 1),
                               ensemble_output)
        hs = RichHiddenState(hs, next_input, ensemble_output)
        return (mean, hs, *optional_uncertainty)

    def process_ensemble_output(
            self, ensemble_output, compute_uncertainty: bool):
        """Compute mean, next input and optionally the uncertainty of the
        ensemble output.

        If the network does not want to propagate the input, `None` should be
        returned instead of `next_input`.

        Returns a 3-tuple:
            (mean, next_input, uncertainty)
        If `compute_uncertainty` is `False`, the implementation may return
        `None` as uncertainty.
        """
        raise NotImplementedError()


class DeterministicPropagatorEnsemble(Ensemble):
    """Ensemble that computes uncertainty as the mean of the standard deviation
    vector computed along the ensemble, assuming that each propagator only
    computes the state and not the uncertainty."""
    def process_ensemble_output(
            self,
            ensemble_output,
            compute_uncertainty: bool):
        """Default implementation of computing the uncertainty of ensemble output."""
        # Convert the whole batch element into a single number.
        mean = ensemble_output.mean(axis=0)
        if compute_uncertainty:
            uncertainty = ensemble_output.var(axis=0)
            compute_mean = lambda x: x.mean(axis=tuple(range(1, x.ndim)))
            if isinstance(uncertainty, TensorCollection):
                # uncertainty = torch.mean(uncertainty.flatten_and_map(compute_mean))
                # This seems to be wrong, it should return a batch, not a single scalar.
                raise NotImplementedError("not tested")
            else:
                uncertainty = compute_mean(uncertainty)
            return (mean, ensemble_output, uncertainty)
        else:
            return (mean, ensemble_output, None)

    def make_trainer(self,
                     config: TrainingConfig,
                     trainer_cls: type = RNNTrainer,
                     **kwargs):
        """Utility function for creating a list of trainers, one for each RNN.

        Arguments:
            config: (TrainingConfig) learning rate and other config
            trainer_cls: (type) trainer class to use for individual propagators
            kwargs: passed to `trainer_cls`
        """
        trainers = []
        for propagator in self.propagators:
            trainer = trainer_cls(propagator, **kwargs,
                                  **config.make(propagator.parameters()))
            trainers.append(trainer)
        return NNEnsembleTrainer(trainers)


def merge_ensemble_of_mu_sigma2(mu, sigma2, compute_sigma2: bool, axis: int = 0):
    """Given (mu, sigma2) outputs of an ensemble, merge them into a single (mu,
    sigma2)."""
    merged_mu = mu.mean(axis=axis)
    if compute_sigma2:
        tmp = sigma2 + mu * mu
        merged_sigma2 = tmp.mean(axis=axis) - merged_mu * merged_mu
    else:
        merged_sigma2 = None
    return (merged_mu, merged_sigma2)


class ProbabilisticPropagatorEnsemble(Ensemble):
    """Computes uncertainty from networks that compute mean and sigma^2.

    Reference:
        Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,
        L. Balaji, A. Pritzel, and C. Blundell,
        Advances in neural information processing systems 30 (2017)
    """
    def process_ensemble_output(
            self,
            ensemble_output,
            compute_uncertainty: bool):
        mu = ensemble_output[..., 0]
        sigma2 = ensemble_output[..., 1]
        mean, uncertainty = merge_ensemble_of_mu_sigma2(
                mu, sigma2, compute_uncertainty)
        if compute_uncertainty:
            def compute_mean(x):
                return x.mean(axis=tuple(range(1, x.ndim)))
            if isinstance(uncertainty, TensorCollection):
                # uncertainty = torch.mean(uncertainty.flatten_and_map(compute_mean))
                # This seems to be wrong, it should return a batch, not a single scalar.
                raise NotImplementedError("not tested")
            else:
                uncertainty = compute_mean(uncertainty)
            return (mean, mu, uncertainty)
        else:
            return (mean, mu, None)

        # TODO: intead of uncertainty, maybe compute confidence, which means normalize this uncertainty to [0, 1] so that we have a more explainable value. This can be done by:
        # conf = something like exp(- sigma^2/range()^2) to normalize
        # TODO: Variational autoencoders - future

    def make_trainer(
            self,
            config: TrainingConfig,
            trainer_cls: type = RNNTrainer,
            loss=None,
            comm=None,
            **kwargs):
        """Utility function for creating a list of trainers, one for each RNN."""
        if loss is None:
            loss = ProbabilisticLosses()
        trainers = []
        for propagator in self.propagators:
            try:
                trainer = trainer_cls(propagator, loss=loss, comm=comm,
                                      **config.make(propagator.parameters()),
                                      **kwargs)
            except Exception as e:
                raise Exception(f"constructing {trainer_cls} failed") from e
            trainers.append(trainer)
        return NNEnsembleTrainer(trainers)
