from adaled.utils.data.datasets import \
        DynamicTrajectoryDataset, CappedTrajectoryDataset, Intracomm, \
        FixedLengthTrajectoryDataset, UniformInMemoryTrajectoryDataset, \
        TrajectoryDatasetWrapper, load_dataset
from adaled.utils.data.replacement import RandomReplacementPolicy

import numpy as np

from typing import Any, Optional, Sequence
import os

__all__ = [
    'CappedFixedLengthTrajectoryDatasetCollection',
    'DynamicTrajectoryDatasetCollection',
]

def _get_capacity(dataset):
    """Determine the capacity of the given dataset.

    Returns `None` is capacity cannot be determined.
    """
    if isinstance(dataset, CappedTrajectoryDataset):
        return dataset.capacity
    elif isinstance(dataset, TrajectoryDatasetWrapper):
        return _get_capacity(dataset.parent)
    else:
        return None


class DynamicTrajectoryDatasetCollection:
    """Dispatch new trajectory batches into training and validation datasets.

    Attributes:
        train_dataset (DynamicTrajectoryDataset): training dataset
        valid_dataset (DynamicTrajectoryDataset): validation dataset
        train_portion (float): Portion of new trajectories that is forwarded to
                       the training dataset. If not set, it is automatically
                       computed from dataset capacities. In that case, if
                       capacities cannot be determined, an error is thrown.
    """
    def __init__(self,
                 train_dataset: DynamicTrajectoryDataset,
                 valid_dataset: DynamicTrajectoryDataset,
                 train_portion: Optional[float] = None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        if train_portion is None:
            train = _get_capacity(train_dataset)
            valid = _get_capacity(valid_dataset)
            if train is None or valid is None:
                raise TypeError(
                        "cannot determine `train_portion` automatically, as "
                        "reading the capacities of the datasets failed")
            train_portion = train / (train + valid)
        self.train_portion = train_portion

    def close(self):
        """Close the training and validation datasets."""
        self.train_dataset.close()
        self.valid_dataset.close()

    def add_trajectories(self, batch: Sequence[Sequence[Any]]):
        """Split the batch according to `self.train_portion` and forward to
        individual datasets."""
        u = np.random.uniform(0.0, 1.0, len(batch))
        to_train_mask = u < self.train_portion
        def apply_mask(mask):
            try:
                return batch[mask]
            except TypeError:
                return [batch[i] for i, m in enumerate(mask) if m]

        self.train_dataset.add_trajectories(apply_mask(to_train_mask))
        self.valid_dataset.add_trajectories(apply_mask(~to_train_mask))

    def save(self, dir: str, comm: Optional[Intracomm] = None, **kwargs):
        """Save the datasets to the given folder."""
        self.train_dataset.save(os.path.join(dir, 'train'), **kwargs)
        self.valid_dataset.save(os.path.join(dir, 'valid'), **kwargs)
        if comm is None or comm.rank == 0:
            import json
            with open(os.path.join(dir, 'metadata.json'), 'w') as f:
                json.dump(self.get_metadata(), f)

    def get_metadata(self):
        return {'train_portion': self.train_portion}

    @classmethod
    def load(cls, dir: str, *args, **kwargs):
        """Load the datasets from the given folder."""
        train_dataset = load_dataset(os.path.join(dir, 'train'), *args, **kwargs)
        valid_dataset = load_dataset(os.path.join(dir, 'valid'), *args, **kwargs)
        import json
        with open(os.path.join(dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        train_portion = float(metadata['train_portion'])

        return cls(train_dataset, valid_dataset, train_portion)


class CappedFixedLengthTrajectoryDatasetCollection(
        DynamicTrajectoryDatasetCollection):
    """A collection of training and validation in-memory dataset capped to
    given capacity and trajectory length.

    This is a convenience class that combines
    :py:class:`.UniformInMemoryTrajectoryDataset`,
    :py:class:`.CappedTrajectoryDataset` and
    :py:class:`.FixedLengthTrajectoryDataset`.
    """

    def __init__(self,
                 train_capacity: int,
                 valid_capacity: int,
                 trajectory_length: int):
        # FIXME: this is too hacky, find a better way to implement
        # save/load mechanism for datasets, something based on
        # state_dict/load_state_dict.
        if not isinstance(train_capacity, int):
            super().__init__(train_capacity, valid_capacity, trajectory_length)
            return

        def _make(capacity: int):
            dataset = UniformInMemoryTrajectoryDataset()
            policy = RandomReplacementPolicy()
            dataset = CappedTrajectoryDataset(dataset, policy, capacity)
            dataset = FixedLengthTrajectoryDataset(dataset, trajectory_length)
            return dataset

        train_dataset = _make(train_capacity)
        valid_dataset = _make(valid_capacity)
        train_portion = train_capacity / (train_capacity + valid_capacity)
        super().__init__(
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                train_portion=train_portion)
