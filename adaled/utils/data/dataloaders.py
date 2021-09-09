from adaled.backends import get_backend, TensorCollection
from adaled.utils.buffer import DynamicArray

import numpy as np
import torch

from typing import Any, Callable, Iterable, Sequence
import math

class _WrapDataLoaderIter:
    __slots__ = ('iter', 'ops')

    def __init__(self, iter, ops):
        self.iter = iter
        self.ops = ops

    def __next__(self):
        out = next(self.iter)
        for op in self.ops:
            out = op(out)
        return out


def make_collate(collate_fn: Callable = torch.utils.data.dataloader.default_collate):
    def collate(batch: Iterable[Any]):
        elem = batch[0]
        if isinstance(elem, TensorCollection):
            return TensorCollection.multimap(
                    lambda parts: get_backend(parts[0]).stack(parts),
                    batch)
        elif isinstance(elem, tuple):
            return tuple(wrapped_collate_fn(items) for items in zip(*batch))
        else:
            return collate_fn(batch)

    return collate

default_collate = make_collate()


class DataLoader(torch.utils.data.DataLoader):
    """Base DataLoader class that recognizes TensorCollection data."""
    def __init__(self,
                 dataset: Sequence[Any],
                 *args,
                 collate_fn: Callable = torch.utils.data.dataloader.default_collate,
                 **kwargs):
        # Simplify the input...
        if isinstance(dataset, DynamicArray):
            dataset = dataset.data

        super().__init__(dataset, *args, collate_fn=make_collate(collate_fn), **kwargs)


class WrappedDataLoader:
    """Wrap a dataset or data loader with transformations."""
    __slots__ = ('loader', 'ops')

    def __init__(self, loader, *ops):
        self.loader = loader
        self.ops = ops

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return _WrapDataLoaderIter(iter(self.loader), self.ops)

    # DataLoader should not be subscriptable!
    # https://stackoverflow.com/questions/61562456/problem-with-dataloader-object-not-subscriptable
    # def __getitem__(self, key):
    #     out = self.loader[key]
    #     for op in self.ops:
    #         out = op(out)
    #     return out
