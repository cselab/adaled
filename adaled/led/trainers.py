from adaled.backends import TensorCollection, cmap
from adaled.led.sampling import SampleCountPolicyConfig, SamplingPolicyConfig
from adaled.nn.ensembles import Ensemble
from adaled.nn.trainers import \
        Trainer, TrainingConfig, NNTrainer, RNNTrainer, set_train
from adaled.solvers import MacroSolver
from adaled.transformers import Transformer, AutoencoderTransformer
from adaled.utils.arrays import join_sequences
from adaled.utils.data.dataloaders import DataLoader, WrappedDataLoader
from adaled.utils.data.datasets import \
        TrajectoryDataset, apply_transformation_on_xF_dataset
from adaled.utils.dataclasses_ import DataclassMixin, dataclass, field
import adaled

import numpy as np
import torch

from typing import Any, Dict, Optional, Sequence, Tuple
import math
import time

_dummy_model = torch.nn.Tanh()
nan64 = np.float64(np.nan)

__all__ = [
    'nan64', 'LEDTrainer', 'SequentialLEDTrainer',
    'SequentialLEDTrainerConfig', 'MacroAndAutoencoder',
    'SimultaneousLEDTrainer',
]

@dataclass
class LEDTrainingOutput(DataclassMixin):
    transformer_train: Optional[Any]
    transformer_valid: Optional[Any]
    macro_train: Any
    macro_valid: Any


@dataclass
class SequentialLEDTrainerConfig(DataclassMixin):
    """
    Arguments:
        macro_batch_size: (int) number of trajectories per batch to use when
                                evaluating or training the macro solver (*)
        transformer_batch_size: (int) number of states per batch to use when
                                evaluating or training the transformer (*)
        transform_in_advance: (bool) transform dataset subset in advance
                              before passing it to the macro trainer

    (*) When the macro solver is trained, the loaded trajectories first have to
    be transformed to the latent space. This step still uses
    `transformer_batch_size` states per transformer batch.
    """
    # TODO: Should these be global and not local batch sizes?
    macro_batch_size: int = 16
    transformer_batch_size: int = 16
    transform_in_advance: bool = True

    states_count_policy: SampleCountPolicyConfig = field(SampleCountPolicyConfig)
    trajectories_count_policy: SampleCountPolicyConfig = field(SampleCountPolicyConfig)
    states_sampling_policy: SamplingPolicyConfig = field(SamplingPolicyConfig)
    trajectories_sampling_policy: SamplingPolicyConfig = field(SamplingPolicyConfig)

    def make(self, *args, **kwargs):
        return SequentialLEDTrainer(self, *args, **kwargs)


class LEDTrainer:
    """Trains macro solver (RNNs) and the transformer (autoencoders) on
    datasets of trajectories."""

    def load_state_dict(self, state: dict):
        raise NotImplementedError()

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def get_hyperparams(self):
        """Get hyperparameters for visualization purposes."""
        return {}

    def compute_losses(self, trajectories: Sequence[TensorCollection]) \
            -> Sequence[TensorCollection]:
        """Compute loss of given trajectories without training the networks.

        Returns transformation and macro solver error, or whichever is
        applicable depending on the training procedure.
        """
        raise NotImplementedError(
                f"compute_losses not implemented for {self.__class__.__name__}. "
                f"If it is only for diagnostics purposes, consider capturing "
                f"this exception and using an empty TensorCollection instead.")

    def train(
            self,
            train_dataset: TrajectoryDataset,
            valid_dataset: Optional[TrajectoryDataset],
            sample_count_policy_kwargs: dict) -> TensorCollection:
        """Train the LED network on a fraction of the given training dataset
        and optionally validate.

        Arguments:
            train_dataset: the training dataset
            valid_dataset: (optional) the validation dataset
            sample_count_policy: (dict) passed to sample count policies

        Returns a collection containing:
            - transformer_train: transformer training loss
            - transformer_valid: transformer validation loss (*)
            - macro_train: macro training loss
            - macro_valid: macro validaiton loss (*).
        (*) if the trainers return `None` for validation loss when `valid_loss`
        is `None`, validation loss is assumed to have the same structure as the
        training loss and is filled with NaNs
        """
        raise NotImplementedError()


class SequentialLEDTrainer(LEDTrainer):
    """Trains autoencoder and RNNs separately."""
    def __init__(self,
                 config: SequentialLEDTrainerConfig,
                 macro_trainer: Trainer,
                 transformer: Transformer,
                 transformer_trainer: Optional[NNTrainer] = None):
        self.config = config
        self.macro_trainer = macro_trainer
        self.transformer_trainer = transformer_trainer
        self.transformer = transformer

        self.states_count_policy = config.states_count_policy.make() \
                if self.transformer_trainer else None
        self.trajectories_count_policy = config.trajectories_count_policy.make()

        self.states_sampling_policy = config.states_sampling_policy.make()
        self.trajectories_sampling_policy = \
                config.trajectories_sampling_policy.make()

    def load_state_dict(self, state: dict):
        if self.transformer_trainer:
            self.transformer_trainer.load_state_dict(state['transformer'])
        if self.macro_trainer:
            self.macro_trainer.load_state_dict(state['macro'])

    def state_dict(self) -> dict:
        return {
            'transformer': self.transformer_trainer.state_dict() \
                    if self.transformer_trainer else None,
            'macro': self.macro_trainer.state_dict() \
                    if self.macro_trainer else None,
        }

    def get_hyperparams(self):
        """Returns a dictionary of transformer and macro trainers, whichever is
        available."""
        out = {}
        if self.transformer_trainer:
            out['transformer'] = self.transformer_trainer.get_hyperparams()
        if self.macro_trainer:
            out['macro'] = self.macro_trainer.get_hyperparams()
        return out

    def compute_losses(self, trajectories: Sequence[TensorCollection]) \
            -> TensorCollection:
        """Returns the transformer loss and the macro prediction loss."""
        out = TensorCollection()
        if not self.transformer_trainer and not self.macro_trainer:
            return out

        transformer_model = \
                self.transformer_trainer.model \
                if self.transformer_trainer \
                else getattr(self.transformer, 'model', _dummy_model)
        with set_train(transformer_model, mode=False), \
                torch.no_grad():
            latent = self.transformer.transform_trajectory_batch(
                    trajectories, self.config.transformer_batch_size)

            if self.transformer_trainer:
                shape = (len(trajectories), len(trajectories[0]))
                states = adaled.merge_axes_01(trajectories['x'])
                latent_states = adaled.merge_axes_01(latent['z'])
                assert len(states) == len(latent_states)
                states_per_batch = self.config.transformer_batch_size
                losses = []
                for i in range(0, len(states), states_per_batch):
                    reconstruct = self.transformer.inverse_transform(
                            latent_states[i : i + states_per_batch])
                    losses.append(self.transformer_trainer.loss(
                            reconstruct, states[i : i + states_per_batch]))
                    del reconstruct   # Free memory.

                losses = join_sequences(losses)
                out['transformer'] = adaled.split_first_axis(losses, shape)

        if self.macro_trainer:
            # FIXME: Don't compute the transformation twice.
            # FIXME: Compute per-state loss instead of broadcasting the
            #        averaged loss. Only if the same can be done for the
            #        training phase.
            losses = self.macro_trainer.compute_losses(latent)
            def broadcast_along_trajectory(a):
                a = adaled.to_numpy(a)
                assert len(a) == len(trajectories)
                a = np.broadcast_to(a, (len(trajectories[0]),) + a.shape)
                return np.moveaxis(a, 0, 1)

            out['macro'] = cmap(broadcast_along_trajectory, losses)

        return out

    def train(self,
              train_dataset: TrajectoryDataset,
              valid_dataset: Optional[TrajectoryDataset],
              sample_count_policy_kwargs: dict) -> TensorCollection:
        t0 = time.time()
        # Train first the AE.
        if self.transformer_trainer:
            num_states = self.states_count_policy(
                    dataset_size=len(train_dataset.as_states()),
                    batch_size=self.config.transformer_batch_size,
                    **sample_count_policy_kwargs)
            train_dataloader, train_indices1 = self.to_transformer_dataloader(
                    train_dataset, is_train=True, num_states=num_states)
            valid_dataloader, valid_indices1 = self.to_transformer_dataloader(
                    valid_dataset, is_train=False)
            train_losses1, valid_losses1 = self.transformer_trainer.train_epoch(
                    train_dataloader, valid_dataloader)
        else:
            train_losses1 = valid_losses1 = None
            train_indices1 = valid_indices1 = None

        # Then the macro solver (RNNs).
        t1 = time.time()
        num_trajectories = self.trajectories_count_policy(
                dataset_size=len(train_dataset.as_trajectories()),
                batch_size=self.config.macro_batch_size,
                **sample_count_policy_kwargs)
        train_dataloader, train_indices2 = self.to_latent_trajectory_dataloader(
                train_dataset, is_train=True, num_trajectories=num_trajectories)
        valid_dataloader, valid_indices2 = self.to_latent_trajectory_dataloader(
                valid_dataset, is_train=False)

        t2 = time.time()
        model = getattr(self.transformer, 'model', _dummy_model)
        with set_train(model, mode=False):
            train_losses2, valid_losses2 = out = self.macro_trainer.train_epoch(
                    train_dataloader, valid_dataloader)

        t3 = time.time()
        losses = LEDTrainingOutput(
                train_losses1, valid_losses1, train_losses2, valid_losses2)
        indices = LEDTrainingOutput(
                train_indices1, valid_indices1, train_indices2, valid_indices2)

        len1a = len(train_indices1) if train_indices1 is not None else 0
        len1b = len(valid_indices1) if valid_indices1 is not None else 0
        len2a = len(train_indices2) if train_indices2 is not None else 0
        len2b = len(valid_indices2) if valid_indices2 is not None else 0

        print(f"Training timings for {len1a}+{len1b} states and "
              f"{len2a}+{len2b} trajectories: transformer={t1 - t0:.6f} "
              f"macro_dataset={t2 - t1:.6f} macro={t3 - t2:.6f} all={t3 - t0:.6f}")
        return losses, indices

    def to_transformer_dataloader(self, dataset, is_train: bool,
                                  num_states: int = -1):
        """Create a data loader that iterates over the single timestep x-states
        of the dataset, ignoring the trajectory structure.

        The batches produced by the data loader are NOT input-output pairs for
        performance reasons, i.e. to ensure we do not construct the same
        batches twice.

        Returns:
            (dataloader over x-states, state indices)
        """
        if dataset is None:
            return None, None
        # Transformers (autoencoders) don't have access to F.
        dataset, indices = self._sample(
                num_states, is_train, dataset.as_states(),
                self.states_sampling_policy, ('trajectory', 'x'))

        dataloader = DataLoader(
                dataset, batch_size=self.config.transformer_batch_size,
                shuffle=False)  # Already shuffled in _sample.
        dataloader = WrappedDataLoader(
                dataloader, lambda batch: cmap(torch.as_tensor, batch))
        return dataloader, indices

    def to_latent_trajectory_dataloader(self, dataset, is_train: bool,
                                        num_trajectories: int = -1):
        """Create a dataloader that iterates over trajectories of the data.

        Returns:
            (dataloader, trajectory indices)
        """
        if dataset is None:
            return None, None
        dataset, indices = self._sample(
                num_trajectories, is_train, dataset.as_trajectories(),
                self.trajectories_sampling_policy, 'trajectory')

        model = getattr(self.transformer, 'model', _dummy_model)
        if self.config.transform_in_advance:
            with torch.no_grad(), \
                    set_train(model, mode=False):
                dataset = apply_transformation_on_xF_dataset(
                        self.transformer, dataset,
                        self.config.transformer_batch_size)

            if len(dataset) > 0:
                dataset = dataset.map(torch.as_tensor)
            dataloader = DataLoader(dataset, batch_size=self.config.macro_batch_size)
        else:
            # Note: not really tested or used any more.

            # If for some reason we don't want to transform the whole partial
            # epoch to the latent space immediately, we can transform it
            # lazily. Note that iterating multiple times through that data
            # loader will trigger multiple transformation of same trajectories.
            def transform(trajectory_batch):
                with torch.no_grad(), \
                        set_train(model, mode=False):
                    out = apply_transformation_on_xF_dataset(
                            self.transformer, trajectory_batch,
                            self.config.transformer_batch_size)
                    return out.map(torch.as_tensor)

            dataloader = DataLoader(dataset, batch_size=self.config.macro_batch_size)
            dataloader = WrappedDataLoader(dataloader, transform)
        return dataloader, indices

    def _sample(self, num_samples, is_train, dataset, policy, key):
        if num_samples == 0 or len(dataset) == 0:
            return [], np.arange(0)
        if num_samples > 0 or is_train:
            indices = policy.sample(dataset, num_samples)
            dataset = torch.utils.data.Subset(dataset[key], indices)
        else:
            dataset = dataset[key]
            indices = np.arange(len(dataset))
        return dataset, indices


class MacroAndAutoencoder(torch.nn.Module):
    """Macro solver network wrapped with the autoencoder networks.

    Helper class for SimultaneousLEDTrainer.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 macro: torch.nn.Module,
                 decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.macro = macro
        self.decoder = decoder

        # This is hacky, here we explictily check for Ensemble, which was
        # designed to be not specific to torch and is hence not a
        # torch.nn.Module. Not being a Module, we have to manually copy the
        # Module attributes (the RNN propagators), in order to include them in
        # self.parameters() for training. We should consider always inheriting
        # from torch.nn.Module.
        if isinstance(macro, Ensemble):
            self.propagators = torch.nn.ModuleList(macro.propagators)

    def forward(self, input, hidden=None):
        # Design notice: for now both encoder and the macro get the external
        # forcing. It is not clear whether the encoder should get it, but the
        # macro must get it, since external forcing may change during the
        # macro-only stage.
        z = self.encoder(input)
        input = TensorCollection(z=z, F=input['F'])
        z_new, hidden = self.macro.advance(input, hidden)
        x_new = self.decoder(z_new)
        return x_new, hidden


class SimultaneousLEDTrainer(LEDTrainer):
    """Trains autoencoder and RNNs simultaneously."""
    def __init__(self,
                 macro: MacroSolver,
                 autoencoder: AutoencoderTransformer,
                 config: TrainingConfig,
                 batch_size: int = 16,
                 trainer_cls: type = RNNTrainer):
        # Pack AE and RNNs into a single module and pass it to an ordinary RNN
        # trainer.
        self.wrapped = MacroAndAutoencoder(
                autoencoder.model.encoder,
                macro,
                autoencoder.model.decoder)
        self.trainer = RNNTrainer(
                self.wrapped,
                **config.make(self.wrapped.parameters()))
        self.batch_size = batch_size

    def train(self,
              train_dataset: TrajectoryDataset,
              valid_dataset: Optional[TrajectoryDataset],
              train_dataset_fraction: float):
        # TODO: preferential selection depending on cmp_error.
        train_dataset = train_dataset.as_trajectories('trajectory')
        if train_dataset_fraction < 1.0:
            train_dataset = random_dataset_subset(
                    train_dataset, fraction=train_dataset_fraction)
            shuffle = False  # Already shuffled.
        else:
            shuffle = True
        train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=shuffle)

        if valid_dataset:
            valid_dataset = valid_dataset.as_trajectories('trajectory')
            valid_dataloader = DataLoader(
                    valid_dataset, batch_size=self.batch_size)
        else:
            valid_dataloader = None

        train_losses, valid_losses = self.trainer.train_epoch(
                train_dataloader, valid_dataloader)

        raise NotImplementedError(
                "should return two TensorCollections, one for losses, one for "
                "indices of states/trajectories used for training")
        # Autoencoder training and validation loss are not defined / not computed.
        return (None, None, train_losses, valid_losses)
