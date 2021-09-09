from adaled.backends import cmap, get_backend, TensorCollection
from adaled.utils.buffer import _EmptyUnknownTypeArray, DynamicArray
from adaled.utils.data.replacement import ReplacementPolicy
import adaled.backends as backends
import adaled.utils.io_ as io_

import numpy as np
import torch

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import math
import os

# FIXME: Either replace add_trajectories with add_trajectory, or base the
# default implementation of add_trajectories on add_trajectory. Inserting
# multiple trajectories at a time complicates the implementation and may cause
# the code to crash when the training is much more expensive than the model
# evaluation, i.e. when too many trajectories are being added in one go.

__all__ = [
    'TrajectoryDataset', 'DynamicTrajectoryDataset',
    'UniformInMemoryTrajectoryDataset', 'TrajectoryDatasetWrapper',
    'FixedLengthTrajectoryDataset', 'CappedTrajectoryDataset', 'load_dataset',
    'apply_transformation_on_dataset', 'apply_transformation_on_xF_dataset',
    'apply_transformation_on_zF_dataset', 'random_dataset_subset',
    'WrappedDataset',
]

Intracomm = 'mpi4py.MPI.Intracomm'

DATASET_DUMP_FILENAME = 'dataset.pt'

_DatasetKey = Union[str, Tuple[str, ...]]

class StatesAccessor:
    """The interface of objects returned by
    :py:meth:`TrajectoryDataset.as_states`."""

    def __len__(self):
        """Return the number of states in the dataset."""
        raise NotImplementedError()

    def __getitem__(self, key):
        """Return a single element or a subset of the dataset."""
        raise NotImplementedError()


class TrajectoriesAccessor:
    """The interface of objects returned by
    :py:meth:`TrajectoryDataset.as_trajectories`."""

    def __len__(self):
        """Return the number of trajectories in the dataset."""
        raise NotImplementedError()

    def __getitem__(self, key):
        """Return a single trajectory or a subset of the dataset."""
        raise NotImplementedError()


class TrajectoryDataset:
    """Trajectory dataset base class."""

    def finalize_load(self, path: str, comm: Optional[Intracomm] = None):
        """Initialize the dataset once the path and the MPI communicator are known."""
        raise NotImplementedError(self)

    def close(self):
        """Close the dataset files (if any). The implementation may assume no
        further API is invoked on this object."""
        pass  # No-op by default.

    def save(self,
             path: str,
             verbose: bool = True,
             hardlink: Optional[bool] = None) -> None:
        """Save the dataset to the given path. Collective operation.

        It is up to the implementation whether the path denotes a file or a
        folder. By default treats the path as a file and stores the whole
        dataset object using :py:meth:`io_.save`.

        Note for implementers: in a distributed dataset setting, saving and
        loading is performed on all ranks, make sure there are no race
        condition and that the pickled data is small.
        """
        io_.save(self, path, verbose=verbose)

    def as_states(self, key: _DatasetKey = ()) -> StatesAccessor:
        """Return a dataset of states.

        Optionally, a string or tuple of strings can be passed an argument, to
        access only certain parts of the collection."""
        raise NotImplementedError(self)

    def as_trajectories(self, key: _DatasetKey = ()) -> TrajectoriesAccessor:
        """Return a dataset of trajectories.

        Optionally, a string or tuple of strings can be passed an argument, to
        access only certain parts of the collection."""
        raise NotImplementedError(self)


class DynamicTrajectoryDataset(TrajectoryDataset):
    """The base class for trajectory datasets that support adding new or replacing
    old trajectories.

    .. note::

        Currently, the dataset API works on batches of trajectories.
        In the future, this might be replaced with API that operates on single
        trajectories, because it might simplify the implementation.
        (Adding batches of trajectories is rare and useful only for small systems.)
    """

    def add_trajectories(self, trajectory_batch: Sequence[Any]) -> None:
        """Add a batch of trajectories to the dataset. Collective operation.

        Depending on the implementation, the trajectories may or may not have
        non-uniform length.
        """
        raise NotImplementedError(self)

    def set_trajectories(self,
                         indices: Sequence[int],
                         trajectories: Sequence[Any]) -> None:
        """Update trajectories at given indices. Collective operation."""
        raise NotImplementedError(self)

    def update_states(self,
                      indices: Sequence[int],
                      key: Tuple[str],
                      values: Sequence[Any]) -> None:
        """Update states with given indices. Local operation."""
        raise NotImplementedError(self)

    def update_trajectories(
            self,
            indices: Sequence[int],
            key: Tuple[str],
            values: Sequence[Any]) -> None:
        """Update trajectories data, without modifying the trajectory length.
        Local operation."""
        raise NotImplementedError(self)


class UniformInMemoryTrajectoryDataset(DynamicTrajectoryDataset):
    """In-memory dynamic dataset of uniformly shaped data.

    Internally, the data is stored as a single :py:class:`adaled.DynamicArray`.
    Depending on the location of the trajectories passed to
    :py:func:`add_trajectories`, the data will be stored either in CPU or in
    GPU memory.
    Useful for simulations with small or moderately large states.

    This dataset's size is unbounded. To limit its capacity, see
    :py:class:`CappedTrajectoryDataset`.

    Does not support nonuniform trajectory length, nonuniform state shape nor distributed training.

    >>> dataset = adaled.UniformInMemoryTrajectoryDataset()
    >>> # One 10-step long trajectory of 4x4 states.
    >>> trajectories = np.zeros((1, 10, 4, 4))
    >>> dataset.add_trajectories(trajectories)
    >>> len(dataset.as_trajectories())
    1
    >>> len(dataset.as_states())
    10
    >>> dataset.as_states()[0]
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    """
    def __init__(self):
        self.dataset = DynamicArray()

    def finalize_load(self, path: str, comm: Intracomm):
        if comm and comm.size > 1:
            # Reading an existing dataset is not an issue, modifying it is.
            raise NotImplementedError("MPI communicators not supported.")

    def as_states(self, key: _DatasetKey = ()):
        data = self.dataset.data
        if isinstance(data, _EmptyUnknownTypeArray):
            return data

        # Merge first two dimensions: trajectory index and timestep index.
        def merge_axes(x):
            shape = x.shape
            # `strict_reshape` asserts that no copy was made.
            return get_backend(x).strict_reshape(x, (shape[0] * shape[1],) + shape[2:])
        return cmap(merge_axes, data)[key]

    def as_trajectories(self, key: _DatasetKey = ()):
        return self.dataset.data[key]

    def add_trajectories(self, trajectory_batch):
        self.dataset.extend(trajectory_batch)

    def set_trajectories(self, indices, batch):
        assert len(indices) == len(batch), (indices, batch.shape)
        if len(indices) > 0:
            self.dataset[indices] = batch

    def update_states(self, indices, key, values):
        if len(indices) > 0:
            self.as_states()[key][indices] = values

    def update_trajectories(self, indices, key, values):
        if len(indices) > 0:
            self.dataset[key][indices] = values


class TrajectoryDatasetWrapper(DynamicTrajectoryDataset):
    """Base wrapper class for trajectory datasets. The original dataset is
    stored as the `parent` attribute.

    This class provides default implementations of :py:func:`as_states`,
    :py:func:`as_trajectories`, :py:func:`update_states` and
    :py:func:`update_trajectories` that simply forward the arguments to the
    parent's corresponding function.

    When storing on disk, the parent is stored independently with a suffix
    `-parent`.
    """
    def __init__(self, parent: DynamicTrajectoryDataset):
        self.parent = parent

    @staticmethod
    def _make_parent_path(path: str):
        a, b = os.path.splitext(path)
        return f'{a}-parent{b}'

    def finalize_load(self, path: str, comm: Optional[Intracomm] = None):
        self.parent = load_dataset(self._make_parent_path(path), comm)

    def save(self, path: str, **kwargs) -> None:
        # Temporarily set parent to None, which is simpler than manipulating
        # __getstate__ and __setstate__, especially for inherited classes.
        parent = self.parent
        self.parent = None
        try:
            super().save(path, **kwargs)
            parent.save(self._make_parent_path(path), **kwargs)
        finally:
            self.parent = parent

    def as_states(self, *args):
        return self.parent.as_states(*args)

    def as_trajectories(self, *args):
        return self.parent.as_trajectories(*args)

    def update_states(self, *args, **kwargs):
        self.parent.update_states(*args, **kwargs)

    def update_trajectories(self, *args, **kwargs):
        self.parent.update_trajectories(*args, **kwargs)


class FixedLengthTrajectoryDataset(TrajectoryDatasetWrapper):
    """Dataset wrapper. Randomly slices trajectories to match the pre-specified
    length before adding them to the parent dataset.
    """

    def __init__(self, parent: DynamicTrajectoryDataset, trajectory_length: int):
        """
        Arguments:
            parent: parent dataset
            trajectory_length: Length to slice to. Trajectories passed to
                    :py:func:`add_trajectories` by the user must be at least
                    `trajectory_length` long.
        """
        super().__init__(parent)
        self.trajectory_length = trajectory_length

    def add_trajectories(self, batch):
        """
        If the trajectories length is larger than the pre-specified trajectory
        length, randomly slice each trajectory before passing to the parent
        dataset.

        Raises an exception if the trajectories are shorter than the
        pre-specified length.

        Arguments:
            batch: an array or a collection of arrays of shape
                   (batch size, trajectory length, state shape...).
        """
        if len(batch) == 0:
            try:
                # Try to preserve the shape. Not really necessary.
                batch = batch[:, :self.trajectory_length]
            except TypeError:
                batch = []
            # Collective operation, forward empty lists as well.
            self.parent.add_trajectories(batch)
            return

        L = self.trajectory_length
        slices = []
        for i, traj in enumerate(batch):
            length = len(traj)
            if length < L:
                raise ValueError(f"trajectory #{i} too short, expected length "
                                 f"of at least {L}, got {length}")

            # Heuristics to determine how many trajectories of length `L` to
            # extract from a trajectory of length `length`. Ideally we would
            # like to avoid fixed length trajectories altogether and instead
            # randomly sample subtrajectories at train time.
            num_slices = (2 * length - L) // L
            start_indices = np.random.randint(0, length - L + 1, num_slices)
            for start in start_indices:
                slices.append(traj[start:start + L])

        def stack(*parts):
            return get_backend(parts[0]).stack(parts)

        new_batch = cmap(stack, *slices)
        self.parent.add_trajectories(new_batch)

    def set_trajectories(self, batch):
        raise NotImplementedError(
                "not supported, put capped trajectory as the "
                "parent of this dataset, not the other way around")


class CappedTrajectoryDataset(TrajectoryDatasetWrapper):
    """Limits a given dynamic dataset to a given capacity.

    If adding new trajectories would exceed the pre-specified capacity, old
    trajectories are replaced instead. In other words, once the parent dataset
    size reaches given capacity, the calls to :py:func:`add_trajectories` are
    replaced with :py:func:`set_trajectories`.
    """
    def __init__(self,
                 parent: DynamicTrajectoryDataset,
                 replacement_policy: ReplacementPolicy,
                 capacity: int,
                 comm: Optional[Intracomm] = None):
        """
        Arguments:
            parent: dataset to wrap
            replacement_policy: policy that determines which old trajectories get replaced by new ones
            capacity: maximum capacity
            comm: MPI communicator
        """
        if comm is None:
            comm = getattr(parent, 'comm', None)
        super().__init__(parent)
        self.capacity = capacity
        self.policy = replacement_policy
        self.comm = comm
        self._all_ranks_full = False
        if comm:
            self._update_all_ranks_full()

    def finalize_load(self, path: str, comm: Intracomm):
        super().finalize_load(path, comm)
        self.comm = comm
        if comm:
            self._update_all_ranks_full()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['comm']
        del state['_all_ranks_full']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.comm = None
        self._all_ranks_full = False

    def _update_all_ranks_full(self):
        """Check if all ranks reached full capacity. This function may be
        invoked only when self.comm is set and when self._all_ranks_full is
        still False."""
        full = len(self.parent.as_trajectories()) >= self.capacity
        send = np.array([int(full)])
        recv = np.array([0])
        self.comm.Allreduce(send, recv)
        self._all_ranks_full = recv[0] == self.comm.size

    def add_trajectories(self, batch: TensorCollection):
        """Add a batch of samples to the dataset.

        If adding the batch would cause the dataset to overflow, overwrite
        existing samples.

        In distributed runs, this function ensures that the parent's
        :py:func:`add_trajectories` is invoked in a collective fashion.
        Ranks that reached the capacity will keep invoking
        :py:func:`add_trajectories` with zero trajectories until every other
        rank reaches its capacity as well.
        """
        num_trajectories = len(batch)
        if num_trajectories > self.capacity:
            raise ValueError(f"attempting to add {len(batch)} trajectories to "
                             f"a dataset with capacity {self.capacity}")
        current = len(self.parent.as_trajectories())
        to_add = max(0, min(num_trajectories, self.capacity - current))

        if to_add or (self.comm and not self._all_ranks_full):
            part = batch[:to_add]
            self.parent.add_trajectories(part)
            self.policy.add_trajectories(part)
            if self.comm and not self._all_ranks_full:
                self._update_all_ranks_full()

        # Always call set_trajectories, since it is a collective operation.
        # Sooner or later, as the dataset is filled, only set_trajectories is
        # invoked anyway.
        part = batch[to_add:]
        indices = self.policy.replace(part)
        self.parent.set_trajectories(indices, part)


def load_dataset(path: str, comm: Optional[Intracomm] = None):
    """Load and return a dataset.

    Invokes dataset partial initialization step `.finalize_load()`, informing
    the dataset of its path and the communicator it is spanned across.
    """
    if os.path.isfile(path):
        load_path = path
    elif os.path.isdir(path):
        # Convention: if a dataset is stored as multiple files in a folder,
        # DATASET_DUMP_FILENAME is the main pickled object.
        load_path = os.path.join(path, DATASET_DUMP_FILENAME)
    else:
        raise FileNotFoundError(f"path `{path}` is neither a file nor a folder")
    # The object itself is pickled, but we pass path and communicator manually
    # since those information is runtime-specified and cannot be stored in the
    # pickle file itself (i.e. we want to support copying/moving datasets as
    # well as non-fixed number of ranks).
    dataset: DynamicTrajectoryDataset = io_.load(load_path)
    dataset.finalize_load(path, comm)
    return dataset


def apply_transformation_on_dataset(
        mapping: Callable,
        dataset: Sequence[Any],
        states_per_batch: int,
        verbose: bool = False):
    """Apply the given transformation on a sequence of trajectories, in
    batches of states.

    Currently, trajectories are assumed to be of equal length.

    Arguments:
        mapping: mapping/transformation to apply, operates on batches of states
        dataset: sequence of trajectories
        states_per_batch: batch size, in number of states
        verbose: (bool) print progress

    Returns:
        concatenated collection-like of transformer outputs
    """
    out = None
    for i, trajectory in enumerate(dataset):
        if verbose:
            print(f"apply transformation: n={len(dataset)}  i={i}", flush=True)
        for j in range(0, len(trajectory), states_per_batch):
            partial = mapping(trajectory[j : j + states_per_batch])

            if out is None:
                # Allocate once the output shape is known. Assuming all
                # trajectories have same length.
                out = backends.extended_emptylike(
                        partial[0], (len(dataset), len(trajectory)))
            out[i, j : j + states_per_batch] = partial

    return out if out is not None else []


def apply_transformation_on_xF_dataset(
        mapping: Callable, *args, **kwargs) -> TensorCollection:
    """Variant of `apply_transformation_on_dataset` that assumes the input has
    structure `{'x': ..., 'F': ...}`, such that the transformation is applied
    only on the `x` part, while the `F` part is kept as-is. Returns a
    `TensorCollection(z=..., F=...)`"""
    def xF_mapping(xF):
        return TensorCollection(z=mapping(xF['x']), F=xF['F'])

    return apply_transformation_on_dataset(xF_mapping, *args, **kwargs)


def apply_transformation_on_zF_dataset(
        mapping: Callable, *args, **kwargs) -> TensorCollection:
    """Variant of `apply_transformation_on_dataset` that assumes the input has
    structure `{'z': ..., 'F': ...}`, such that the (inverse) transformation is
    applied only on the `z` part, while the `F` part is kept as-is. Returns a
    `TensorCollection(x=..., F=...)`"""
    def zF_mapping(zF):
        return TensorCollection(x=mapping(zF['z']), F=zF['F'])

    return apply_transformation_on_dataset(zF_mapping, *args, **kwargs)


def random_dataset_subset(
        dataset, *,
        fraction: Optional[float] = None,
        count: Optional[int] = None,
        lazy: bool = True):
    """Create a random shuffled subset of the given dataset.

    Arguments:
        dataset: the original dataset
        fraction: the fraction of the dataset to keep (*)
        count: the number of elements to keep (*)
        lazy: return as a lazy torch Subset class, do not apply dataset[indices]

    (*) exactly one argument expected
    """
    if (fraction is None) + (count is None) != 1:
        raise TypeError("expected either `fraction` or `count`")
    length = len(dataset)
    if fraction is not None:
        count = math.ceil(length * fraction)
    # In principle we could put `replace=True` to optimize computing the random
    # indices. Also, not sure if this should be np or torch.
    indices = np.random.choice(length, count, replace=False)
    if lazy:
        return torch.utils.data.Subset(dataset, indices)
    else:
        return [dataset[idx] for idx in indices]


class _WrapDatasetIter:
    __slots__ = ('iter', 'ops')

    def __init__(self, iter, ops):
        self.iter = iter
        self.ops = ops

    def __next__(self):
        out = next(self.iter)
        for op in self.ops:
            out = op(out)
        return out


class WrappedDataset:
    """Wrap a dataset or data loader with transformations."""
    __slots__ = ('dataset', 'ops')

    def __init__(self, dataset, *ops):
        self.dataset = dataset
        self.ops = ops

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return _WrapDatasetIter(iter(self.dataset), self.ops)

    def __getitem__(self, key):
        out = self.dataset[key]
        for op in self.ops:
            out = op(out)
        return out
