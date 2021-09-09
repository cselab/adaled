from adaled.backends import TensorCollection, cmap, get_backend
from adaled.backends.backend_torch import TorchBackend
from adaled.nn.loss import MSELosses
from adaled.utils.buffer import DynamicArray
from adaled.utils.dataclasses_ import DataclassMixin, dataclass, field
import adaled

import numpy as np
import torch

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
import math

hvd = None  # Lazy load.

Losses = Callable[[Sequence[Any], Sequence[Any]], Sequence[Any]]

def _load_horovod():
    global hvd
    try:
        import horovod as hvd
    except ImportError:
        raise Exception("horovod required for distributed training")
    import horovod.torch


@dataclass
class TrainingConfig(DataclassMixin):
    """Specification of optimizer and scheduler."""
    optimizer: str = 'adam'
    lr: float = 0.001
    weight_decay: float = 0.0
    optimizer_kwargs: Dict[str, Any] = field(dict)

    scheduler: str = 'none'
    scheduler_kwargs: Dict[str, Any] = field(dict)

    def make(self, trainable_params) -> Dict[str, Any]:
        optimizer = make_optimizer(
                self.optimizer, trainable_params, lr=self.lr,
                weight_decay=self.weight_decay, **self.optimizer_kwargs)
        scheduler = make_scheduler(self.scheduler, optimizer,
                                   **self.scheduler_kwargs)
        return {'optimizer': optimizer, 'scheduler': scheduler}


def make_optimizer(name, trainable_params, **kwargs):
    if name == 'adam':
        return torch.optim.Adam(trainable_params, **kwargs)
    elif name == 'sgd':
        kwargs.setdefault('momentum', 0.9)
        return torch.optim.SGD(trainable_params, **kwargs)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(trainable_params, **kwargs)
    elif name == 'adabelief':
        from adabelief_pytorch import AdaBelief
        kwargs.setdefault('eps', 1e-16)
        kwargs.setdefault('betas', (0.9, 0.999))
        kwargs.setdefault('weight_decouple', True)
        kwargs.setdefault('rectify', False)
        return AdaBelief(trainable_params, **kwargs)
    else:
        raise ValueError(f"unrecognized optimizer '{name}'")


def make_scheduler(name, optimizer, **kwargs):
    if name == 'none':
        return None
    elif name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"unrecognized scheduler '{name}'")


class set_train:
    """Context manager for setting the torch mode to training or evaluation."""
    def __init__(self, model, mode=True):
        self.model = model
        self.was_training = model.training
        model.train(mode)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.was_training)


def evaluate_rnn_on_trajectory(cell, input, h=None):
    """
    Evaluate the RNN in a teacher-forcing fashion, where the network is fed one
    timestep at a time with (only) the hidden state from the previous step.

    Arguments:
        cell: RNN cell that operates on tensors of shape (batch size, ...)
        input: tensor (batch size, trajectory length, ...)

    Output: tensor (batch size, trajectory length, ...)
    """
    trajectory_length = len(input[0])
    trajectory = None
    for i in range(trajectory_length):
        output, h = cell(input[:, i], h)
        if i == 0:
            # Allocate the output array once the output shape is known and
            # store all results directly there, without first creating the list
            # of all outputs and then stacking them together. This should save
            # memory usage approx. by 1/2, although it is not clear if it helps
            # or is counterproductive when gradients are computed.
            trajectory = get_backend(output).empty(
                    output.shape[:1]  + (trajectory_length,) + output.shape[1:])
        trajectory[:, i] = output
    return trajectory, h


def get_zero_loss(loss: Callable, backend=torch):
    """Try to call `loss.get_zero_loss(...)`. If this fails, return a zero 0D
    tensor."""
    if hasattr(loss, 'get_zero_loss'):
        return loss.get_zero_loss(backend)
    else:
        return backend.zeros(())


class Trainer:
    """Base trainer class."""
    def load_state_dict(self, state: dict):
        raise NotImplementedError()

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def get_hyperparams(self):
        """Get hyperparameters for visualization purposes."""
        return {}

    def compute_losses(self, inputs: Iterable[Any]) -> Sequence[Any]:
        """Compute the loss for each input, without affecting the networks."""
        raise NotImplementedError(self)

    def train_epoch(
            self,
            train_loader: Iterable[Any],
            valid_loader: Optional[Iterable[Any]] = None) \
                    -> Tuple[Sequence[float], Optional[Sequence[float]]]:
        """Train the network on the given training dataset for one epoch. and,
        optionally, validate on a validation set.

        Returns a pair
            (training losses, optional validation losses)
        with one loss for each element of the dataset.
        """
        raise NotImplementedError(self)


class NNTrainer(Trainer):
    """Neural network trainer.

    Assumes that the data loader returns a tuple (input_batch, target_batch).
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer=None,
                 loss: Losses = None,
                 scheduler=None,
                 comm=None):
        """
        Arguments:
            optimizer: optimizer to use, Adam by default
            loss: loss function, PerItemMSELoss by default, expected to
                  compute the loss for each element of the batch (not the
                  mean over the batch!)
            scheduler: learning rate scheduler
            comm: (optional, MPI communicator) if passed, the optimizer will be
                  wrapped with horovod.torch.DistributedOptimizer
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        if comm and comm.size > 1:
            _load_horovod()
            optimizer = hvd.torch.DistributedOptimizer(
                    optimizer, named_parameters=model.named_parameters())
        if loss is None:
            loss = MSELosses()

        self.model = model
        self.optimizer = optimizer
        self.loss = loss  # Should this be renamed to `losses` for clarity?
        self.scheduler = scheduler
        self.comm = comm

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state['optimizer'])
        if state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])

    def state_dict(self) -> dict:
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() \
                    if hasattr(self.scheduler, 'state_dict') else None,
        }

    def get_hyperparams(self):
        """Return learning rates of each parameter group (usually only one)."""
        return {'lr': [group['lr'] for group in self.optimizer.param_groups]}

    def compute_losses(self, inputs: Sequence[Any]):
        with set_train(self.model, mode=False), \
                torch.no_grad():
            input_batch, target_batch = next(self.postprocess_loader([inputs]))
            output_batch = self.evaluate_batch(input_batch)
            losses = self.loss(output_batch, target_batch)
            assert len(losses) == len(output_batch)
        return losses

    def train_epoch(self, train_loader, valid_loader=None):
        with set_train(self.model):
            train_losses = self._train_epoch(train_loader)
        if valid_loader is not None:
            valid_losses = self._evaluate_validation(valid_loader)
            if self.scheduler:
                self.scheduler_step(self.scheduler, valid_losses)
        else:
            valid_losses = None
        return train_losses, valid_losses

    def scheduler_step(self, scheduler, local_valid_losses, needs_loss=False):
        """Invoke scheduler.step() with the optional validation loss argument.

        In the case of ReduceLROnPlateau scheduler, computes the global
        validation loss and passes it to .step(). Otherwise, .step() is invoked
        with no arguments.

        Override for if custom behavior is needed.
        """
        needs_loss = needs_loss or isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        if needs_loss:
            valid_loss = local_valid_losses.mean()
            if isinstance(valid_loss, TensorCollection):
                global_valid_loss = sum(valid_loss.allvalues())
            else:
                global_valid_loss = valid_loss
            if self.comm and self.comm.size > 1:
                # Here we assume that the scheduler is deterministic and
                # that all ranks will schedule the same learning rate.
                global_valid_loss = self.comm.allreduce(
                        adaled.to_numpy(global_valid_loss))
            scheduler.step(global_valid_loss)
        else:
            scheduler.step()

    def postprocess_loader(self, dataloader):
        """Perform any required postprocessing of the data loader.

        This function exists for optimization purposes, when performing data
        transformation on the dataset would cause repeated operations."""
        return dataloader

    def _train_epoch(self, train_loader):
        """Perform one epoch of training. Returns the average loss."""
        assert self.model.training
        assert torch.is_grad_enabled()
        # TODO: Reserve memory in advance. Keep track of latest length?
        all_losses = DynamicArray(like=get_zero_loss(self.loss))
        for input_batch, target_batch in self.postprocess_loader(train_loader):
            # Torch documentation recommends for performance reasons to reset
            # gradients to None instead of zeroing them. However, horovod
            # doesn't support that.
            if self.comm and self.comm.size > 1:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)

            output = self.evaluate_batch(input_batch)
            losses = self.loss(output, target_batch)
            self._losses_backward(losses)
            self.optimizer.step()

            assert len(losses) == len(output), \
                    "loss func is expected to return a batch of losses"
            all_losses.extend(losses.detach())

        if self.comm and self.comm.size > 1:
            hvd.torch.join()

        return all_losses.data

    def _losses_backward(self, tensor: Union[torch.Tensor, TensorCollection]):
        tensor = tensor.mean()  # Mean over the batch.
        if isinstance(tensor, TensorCollection):
            tensor = sum(tensor.allvalues())
        tensor.backward()

    def evaluate_batch(self, batch):
        """Evaluate a single batch of inputs on the model.

        Overloadable. See e.g. RNNTrainer.evaluate_batch."""
        return self.model(batch)

    def _evaluate_validation(self, validation_loader):
        """Evaluate the model on the validation set (with gradients computation
        turned off). Returns the validation error."""
        # TODO: Reserve memory in advance. Keep track of latest length?
        all_losses = DynamicArray(like=get_zero_loss(self.loss))
        with set_train(self.model, mode=False):
            with torch.no_grad():
                for input_batch, target_batch in \
                        self.postprocess_loader(validation_loader):
                    output = self.evaluate_batch(input_batch)
                    losses = self.loss(output, target_batch)
                    assert len(losses) == len(output), \
                            "loss func is expected to return a batch of losses"
                    all_losses.extend(losses)
        return all_losses.data


class AutoencoderTrainer(NNTrainer):
    """Trainer that trains the network to reproduce the input."""
    def postprocess_loader(self, dataloader):
        """Returns a dataloader that prepares an input-output batch pair from a
        state batch. Converts `x` to an `(x, x)` pair."""
        for batch in dataloader:
            yield (batch, batch)


class RNNTrainer(NNTrainer):
    """Trainer that trains the RNN for one-step predictions."""
    # TODO: parameter whether it is real or latent space trainer, add assert

    def postprocess_loader(self, dataloader):
        """Returns a dataloader that prepares an input-output batch pair from a
        trajectory batch.

        Converts a batch b={x/z=..., F=...} into a pair (b[:, :-1], x/z[:, 1:]).
        (Works for both latent and real space.)
        """
        for batch in dataloader:
            if 'z' in batch:
                yield (batch[:, :-1], batch['z'][:, 1:])
            else:
                yield (batch[:, :-1], batch['x'][:, 1:])

    def evaluate_batch(self, batch):
        """Evaluate a single batch of trajectories on the model, in a
        teacher-forcing fashion.

        Arguments:
            batch: a tensor (batch size, trajectory size, state shape...)

        Output: tensor (batch size, trajectory size, state shape...)
        """
        # The trainer is interested only in the output, not the hidden state.
        output, h = evaluate_rnn_on_trajectory(self.model, batch)
        return output


class NNTrainerWithAdversarialLoss(NNTrainer):
    """For each input, evaluates the loss function on the input itself and on
    an adversarial input created by shifting the input in the direction of the
    gradient of loss with respect to the input.

    Arguments:
        adversarial_eps: (float) shift magnitude, defaults to 0.01
        adversarial_loss: (callable) loss to apply on the adversarial input,
                          defaults to `self.loss`

    Reference:
        Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,
        L. Balaji, A. Pritzel, and C. Blundell,
        Advances in neural information processing systems 30 (2017)
    """
    def __init__(self,
                 *args,
                 adversarial_eps: float = 0.01,
                 adversarial_loss: Optional[Losses] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if adversarial_loss is None:
            adversarial_loss = self.loss
        self.adversarial_eps = adversarial_eps
        self.adversarial_loss = adversarial_loss

    def train_epoch(self, train_loader, valid_loader=None):
        # Override the default implementation to fix the structure of the
        # validation loss in case `valid_loader` is `None`. Namely, the
        # diagnostics code assumes that validation loss by default has the same
        # structure as the training loss, in case the loss is a
        # `TensorCollection`. Here, however, the validation loss contains only
        # the "original" part and no "adversarial" part.
        train_losses, valid_losses = \
                super().train_epoch(train_loader, valid_loader)
        if valid_losses is None:
            valid_losses = np.nan * train_losses['original']
        return train_losses, valid_losses

    def _train_epoch(self, train_loader):
        """Perform one epoch of training. Returns the average loss."""
        assert self.model.training
        assert torch.is_grad_enabled()
        # TODO: Reserve memory in advance to reduce the number of allocations.
        #       Keep track of latest length?
        all_losses1 = DynamicArray(like=get_zero_loss(self.loss))
        all_losses2 = DynamicArray(like=get_zero_loss(self.adversarial_loss))
        for input_batch, target_batch in self.postprocess_loader(train_loader):
            # Torch documentation recommends for performance reasons to reset
            # gradients to None instead of zeroing them. However, horovod
            # doesn't support that.
            if self.comm and self.comm.size > 1:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)

            # Gradient required for computing the adversarial input.
            input_z = input_batch['z']
            input_z.requires_grad = True

            # Compute the loss (with the gradient) on the original input.
            # Note that the gradient is on per-input basis, so no
            # synchronization is needed for distributed simulations.
            output = self.evaluate_batch(input_batch)
            losses1 = self.loss(output, target_batch)
            assert len(losses1) == len(target_batch)
            all_losses1.extend(losses1.detach())
            self._losses_backward(losses1)

            # Compute the loss on the adversarial input.
            adversarial_input_batch = self._make_adversarial_input(input_batch)
            input_z.requires_grad = False
            input_z.grad = None
            output = self.evaluate_batch(adversarial_input_batch)
            losses2 = self.adversarial_loss(output, target_batch)
            assert len(losses2) == len(target_batch)
            all_losses2.extend(losses2.detach())
            self._losses_backward(losses2)

            self.optimizer.step()

        if self.comm and self.comm.size > 1:
            hvd.torch.join()

        return TensorCollection(original=all_losses1.data,
                                adversarial=all_losses2.data)

    def _make_adversarial_input(self, input_batch):
        assert len(input_batch.keys()) == 2  # z and F only
        z = input_batch['z']
        z = z + self.adversarial_eps * torch.sign(z.grad)
        return TensorCollection(z=z, F=input_batch['F'])


class RNNTrainerWithAdversarialLoss(NNTrainerWithAdversarialLoss, RNNTrainer):
    pass


class NNEnsembleTrainer(Trainer):
    """A compound trainer, used for training e.g. an ensemble of networks."""
    def __init__(self, trainers):
        self.trainers = trainers

    def load_state_dict(self, state: dict):
        assert len(state) == len(self.trainers)
        for i, trainer in enumerate(self.trainers):
            trainer.load_state_dict(state[f'trainer{i}'])

    def state_dict(self) -> dict:
        return {
            f'trainer{i}': trainer.state_dict()
            for i, trainer in enumerate(self.trainers)
        }

    def get_hyperparams(self):
        """Return learning rates of all optimizers and their parameter groups."""
        lr = [
            [group['lr'] for group in trainer.optimizer.param_groups]
            for trainer in self.trainers
        ]
        return {'lr': lr}

    def compute_losses(self, inputs):
        """Return losses averaged along the ensemble."""
        losses = [trainer.compute_losses(inputs) for trainer in self.trainers]
        def mean(*args):
            return get_backend(args[0]).stack(args, axis=0).mean(0)

        losses = cmap(mean, *losses)
        return losses

    def train_epoch(self, *args, **kwargs):
        """Propagate the training and validation data loaders and other
        parameters to each ensemble trainer individually.

        The training and validation losses are averaged along the ensemble.
        """
        # NOTE: This implementation is suboptimal, as it iterates over the
        # dataset N times, where N is the ensemble size. Depending on the batch
        # size, trajectory size and latent state size, it may be beneficial to
        # iterate batch by batch instead of trainer by trainer. This is
        # especially important when the dataloader or dataset are performing
        # transformations on the fly.
        # Note that performing batch by batch must be done carefully in a way
        # that won't affect the LR scheduler.
        # Alternatively, we should use one trainer for the whole ensemble.

        total_train_losses = 0.0
        total_valid_losses = None
        for trainer in self.trainers:
            train_losses, valid_losses = trainer.train_epoch(*args, **kwargs)
            total_train_losses += train_losses
            if valid_losses is not None:
                if total_valid_losses is None:
                    total_valid_losses = 0.0
                total_valid_losses += valid_losses
        mean_train_losses = total_train_losses / len(self.trainers)
        if total_valid_losses is not None:
            mean_valid_losses = total_valid_losses / len(self.trainers)
        else:
            mean_valid_losses = None
        return mean_train_losses, mean_valid_losses
