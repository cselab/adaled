from adaled.utils.buffer import DynamicArray

import numpy as np

from typing import Optional, Sequence, Union

class ReplacementPolicy:
    """Defines which old trajectories will be overwritten when new trajectories
    are added to full datasets."""

    def add_trajectories(self, trajectory_batch):
        """Invoked by the dataset when it is not yet full."""
        raise NotImplementedError()

    def replace(self, trajectory_batch) -> Sequence[int]:
        """Invoked by the dataset when its capacity is reached.

        Returns an array of indices of old trajectories to replace.
        """
        raise NotImplementedError()


class RandomReplacementPolicy(ReplacementPolicy):
    """Replace randomly selected old elements."""
    def __init__(self,
                 rng: Optional[Union[np.random.Generator, int]] = None):
        if rng is None or isinstance(rng, (np.integer, int)):
            rng = np.random.default_rng(rng)
        self.size = 0
        self.rng = rng

    def add_trajectories(self, batch):
        self.size += len(batch)

    def replace(self, batch):
        assert len(batch) <= self.size
        return self.rng.choice(self.size, len(batch), replace=False)
