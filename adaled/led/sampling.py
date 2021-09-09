from adaled.backends import TensorCollection
from adaled.utils.dataclasses_ import DataclassMixin, dataclass, field

import numpy as np
import torch

from typing import Any, Dict, Optional, Tuple, Sequence
import copy
import math
import warnings

__all__ = [
    'SampleCountPolicyConfig', 'SampleCountPolicy', 'SamplingPolicyConfig',
    'SamplingPolicy', 'RandomSamplingPolicy', 'PreferentialSamplingPolicy',
]


@dataclass
class SampleCountPolicyConfig(DataclassMixin):
    """
    Determines the number of samples (states for the transformer, trajectories
    for the macro solver) to use for training in a given partial epoch.

    Available models:
        fixed: the number of samples is fixed, or limited by the currently
                available number of samples
        per_timestep: each server trains on average `per_timestep` samples per
                simulation time step it received from clients
        fraction: use a fixed fraction of the currently available dataset
    """
    fixed: Optional[int] = None
    per_timestep: Optional[float] = None
    fraction: Optional[float] = None
    floor_to_batch_size: bool = True

    def make(self):
        num_set = (self.fixed is not None) \
                + (self.per_timestep is not None) \
                + (self.fraction is not None)
        if num_set == 0:
            warnings.warn("Sample count policy not set, assuming fraction=0.10.")
            self = copy.copy(self)
            self.fraction = 0.10
        elif num_set > 1:
            raise TypeError(f"expected at most one non-None entry: {self}")
        return SampleCountPolicy(self)


class SampleCountPolicy:
    def __init__(self, config: SampleCountPolicyConfig):
        self.config = config
        self._total_count = 0

    def __call__(
            self,
            dataset_size: int,
            batch_size: int,
            total_client_timesteps: int) -> int:
        """
        Arguments:
            total_client_timesteps: (int) total number of timesteps (of all
                    stages) for all cycles that this server rank received from
                    its clients
        """
        config = self.config
        if config.fixed is not None:
            out = config.fixed
        elif config.per_timestep is not None:
            out = int(total_client_timesteps * config.per_timestep - self._total_count)
            out = max(0, out)
        elif config.fraction is not None:
            out = math.ceil(dataset_size * config.fraction)

        out = min(out, dataset_size)
        if config.floor_to_batch_size and out > batch_size:
            out = out // batch_size * batch_size

        self._total_count += out
        return out


@dataclass
class SamplingPolicyConfig(DataclassMixin):
    type: str = 'random'
    kwargs: Dict[str, Any] = field(dict)

    def make(self) -> 'SamplingPolicy':
        if self.type == 'random':
            return RandomSamplingPolicy()
        elif self.type == 'preferential':
            return PreferentialSamplingPolicy(**self.kwargs)
        else:
            raise ValueError(f"unrecognized sampling policy type: {self.type}")


class SamplingPolicy:
    def sample(self, dataset: Sequence[TensorCollection], n: int) -> Sequence[int]:
        raise NotImplementedError(self)


class RandomSamplingPolicy(SamplingPolicy):
    def sample(self, dataset: Sequence[TensorCollection], n: int):
        indices = np.random.choice(len(dataset), n, replace=False)
        return indices


class PreferentialSamplingPolicy(SamplingPolicy):
    """Sample trajectories according to the following weight:
        w(s) = max(a, b + log(s[key]))
    """
    def __init__(self, key: Tuple[str], a: float, b: float):
        self.key = key
        self.a = a
        self.b = b

    def sample(self, dataset: Sequence[TensorCollection], n: int):
        w = np.maximum(self.a, self.b + np.log(dataset[self.key][:]))
        w *= 1 / w.sum()
        indices = np.random.choice(len(dataset), n, p=w, replace=False)
        return indices
