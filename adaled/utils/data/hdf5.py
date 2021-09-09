from adaled.backends import TensorCollection
from adaled.utils.buffer import DynamicArray
from adaled.utils.data.datasets import \
        DATASET_DUMP_FILENAME, _DatasetKey, DynamicTrajectoryDataset, Intracomm
from adaled.utils.data.dataloaders import default_collate
from adaled.utils.misc import exscan_with_total
from adaled.utils.io_hdf5 import H5PyFile, load_hdf5_group
import adaled
import adaled.utils.io_ as io_

import h5py
import numpy as np

from typing import Any, List, Optional, Sequence, Tuple, Union
import os
import re
import sys

# FIXME: The existing content of the HDF5 file must be ignored (or even erased)
# when starting a new run from scratch, because the replacement policy
# information is not stored.

_KeySimple = Union[str, int]
_Key = Union[Tuple[_KeySimple, ...], _KeySimple]

def _extract_all_trajectory_groups(f: H5PyFile, fmt: str, regex: str) \
        -> List[Tuple[int, h5py.Group]]:
    """Find all top-level groups that match the given regular expression
    `regex`. Verify that they match the original trajectory format `fmt`.

    Returns a list of pairs (global trajectory index, group), sorted by the
    global trajectory index.
    """
    out = []
    regex = re.compile(regex)
    for name, group in f.items():
        match = regex.match(name)
        if match:
            idx = int(match.group(1))
            if name.strip('/') != fmt.format(idx).strip('/'):
                raise ValueError("inconsistent fmt and regex", (name, fmt, fmt.format(idx)))
            out.append((idx, group))

    # Just in case, sort to ensure every rank has the same ordering.
    out.sort(key=lambda row: row[0])
    return out


def _key_to_h5path(key: _DatasetKey):
    if isinstance(key, str):
        return '/' + key
    elif isinstance(key, tuple):
        return '/' + '/'.join(key)
    else:
        raise TypeError(f"unrecognized key: {key}")


def _handle_key(accessor, key: Union[str, Tuple[str, ...]]):
    if not isinstance(key, tuple):
        key = (key,)
    paths = [k for k in key if isinstance(k, str)]
    key = [k for k in key if not isinstance(k, str)]
    if paths:
        accessor = accessor.__class__(
                accessor.dataset, accessor.path + '/' + '/'.join(paths))
    if not key:
        return accessor, None
    assert len(key) == 1, "more than one non-string index not implemented"
    key = key[0]
    n = len(accessor)
    if key < -n or key >= n:
        raise IndexError(key)
    if key < 0:
        key += n
    return accessor, key


def _trajectory_length(group: h5py.Group) -> int:
    """Return the length of the given trajectory by finding any dataset and
    returning its length. Throws an exception if the group is empty."""
    dataset = None
    def visitor(name, obj):
        nonlocal dataset
        if isinstance(obj, h5py.Dataset):
            dataset = obj

    group.visititems(visitor)

    if dataset is None:
        raise RuntimeError(f"no dataset found in group {group}")

    return len(dataset)


class _HDF5StatesAccessor:
    """Helper class for handling .as_states()."""
    __slots__ = ('dataset', 'path')

    def __init__(self, dataset: 'HDF5DynamicTrajectoryDataset', path: str = '/'):
        self.dataset = dataset
        self.path = path  # Dataset path.

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, path={self.path!r})"

    def _check(self):
        dataset = self.dataset
        return dataset

    def __len__(self):
        """Return the length of the current rank's dataset subset."""
        dataset = self._check()
        return dataset._get_local_offsets()[-1]

    def __getitem__(self, key: _Key) \
            -> Union['_HDF5StatesAccessor', TensorCollection, np.ndarray]:
        self, key = _handle_key(self, key)
        if key is None:
            return self
        traj_local_index, timestep_index = self.dataset._local_to_traj_timestep(key)
        try:
            def loader(dataset):
                return dataset[timestep_index]

            group = self.dataset._get_local(self.path, traj_local_index)
            if isinstance(group, h5py.Group):
                return load_hdf5_group(group, loader)
            else:
                return loader(group)
        except IndexError as e:
            raise RuntimeError("loading error") from e


class _HDF5TrajectoriesAccessor:
    """Helper class for handling .as_trajectories()."""
    __slots__ = ('dataset', 'path')

    def __init__(self, dataset: 'HDF5DynamicTrajectoryDataset', path: str = '/'):
        self.dataset = dataset
        self.path = path

    def __len__(self):
        return len(self.dataset._local_lengths)

    def __getitem__(self, key: _Key) \
            -> Union['_HDF5TrajectoriesAccessor', TensorCollection, np.ndarray]:
        self, key = _handle_key(self, key)
        if key is None:
            return self
        obj = self.dataset._get_local(self.path, key)
        return load_hdf5_group(obj) if isinstance(obj, h5py.Group) else obj


class HDF5DynamicTrajectoryDataset(DynamicTrajectoryDataset):
    """Distributed dynamic HDF5 dataset with a single `.h5` file for storage.

    Each trajectory is stored as one top-level group, defined by the `traj_fmt`
    argument. Each trajectory is assumed to be a :py:class:`.TensorCollection`.
    One dataset is created for each such tensor, according to the collection
    hierarchy.

    Internally, when a communicator is given, the `mpio` HDF5 driver is used,
    which requires that all changes to the metadata are done in a collective
    fashion. In other words, the implementation requires that
    :py:func:`add_trajectories` and :py:func:`set_trajectories` are invoked
    collectively.
    """
    size_type = np.int32

    def __init__(self,
                 dir: str,
                 filename: str,
                 comm: Optional[Intracomm] = None,
                 traj_fmt: str = '/traj-{:07d}',
                 traj_fmt_re: str = r'/?traj-(\d+)'):
        """
        Arguments:
            dir: the storage folder
            filename: the file name
            comm: (optional) MPI communicator
            traj_fmt: format of the HDF5 top-level group name for trajectories
            traj_fmt_re: a regex matching `traj_fmt`
        """
        if int(re.match(traj_fmt_re, traj_fmt.format(123)).group(1)) != 123:
            raise ValueError(f"`traj_fmt`={traj_fmt!r} and "
                             f"`traj_fmt_re`={traj_fmt_re!r} are inconsistent")
        super().__init__()
        self.filename = filename
        self.traj_fmt = traj_fmt
        self.traj_fmt_re = traj_fmt_re

        self.comm = None
        self.dir = None
        self.path = None  # For testing, do not remove.
        self.finalize_load(dir, comm)

        # self.verbose_file = sys.stderr
        self.verbose_file = os.devnull

    def __repr__(self):
        return f"{self.__class__.__name__}(dir={self.dir!r}, ...)"

    def __getstate__(self):
        assert self.comm is None or self.comm.rank == 0
        state = self.__dict__.copy()
        del state['_local_lengths']      # Cache, depends on comm.
        del state['_local_offsets']      # Cache, depends on _local_lengths.
        del state['_local_to_global']    # Depends on comm.
        del state['_next_global_index']  # Cache.
        del state['comm']                # Runtime property.
        del state['dir']                 # Runtime property.
        del state['f']                   # Runtime property.
        del state['path']                # Runtime property.
        del state['verbose_file']        # Runtime property.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.comm = None
        self.dir = None
        self.verbose_file = os.devnull

    def finalize_load(self, dir: str, comm: Optional[Intracomm] = None):
        assert self.comm is None, "already initialized"
        assert self.dir is None, "already initialized"
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.comm = comm
        self.path = os.path.join(dir, self.filename)
        kwargs = {'comm': comm, 'driver': 'mpio'} if comm else {}
        # Note: for some reason, HDF5 sometimes says "cannot open file, file
        # exists" here even though the mode is 'a'. Changing mode to 'w' is not
        # a solution because it deletes the dataset and therefore breaks
        # `load_dataset`.
        self.f = H5PyFile(self.path, 'a', **kwargs)

        # Read the metadata (index and length) of all available trajectories
        # and split among the ranks. Note that the set of all global trajectory
        # indices is not necessarily contiguous.
        comm_size = comm.size if comm else 1
        comm_rank = comm.rank if comm else 0
        groups = _extract_all_trajectory_groups(self.f, self.traj_fmt, self.traj_fmt_re)
        self._next_global_index = \
                1 + max(idx for idx, group in groups) if groups else 0

        groups = groups[comm_rank::comm_size]  # Distribute.
        self._local_lengths = DynamicArray(like=self.size_type(123))
        self._local_lengths.extend(
                [_trajectory_length(group) for idx, group in groups])
        self._local_to_global = DynamicArray(like=self.size_type(123))
        self._local_to_global.extend([idx for idx, group in groups])
        self._local_offsets = None

    def close(self):
        self.f.close()
        self.f = None

    def save(self, dir: str, hardlink: bool = False, **kwargs):
        """Save the metadata to a pickle file and copy the HDF5 file."""
        # Save the pickled file on the first rank.
        comm = self.comm
        if not comm or comm.rank == 0:
            super().save(os.path.join(dir, DATASET_DUMP_FILENAME), **kwargs)

        self.f.flush()
        if comm:
            comm.Barrier()  # Just in case.
        if not comm or comm.rank == 0:
            src = os.path.join(self.dir, self.filename)
            dst = os.path.join(dir, self.filename)
            if hardlink:
                os.link(src, dst)
            else:
                import shutil
                shutil.copy2(src, dst)
        if comm:
            comm.Barrier()

    def _get_local_offsets(self):
        """Return the trajectory length prefix sum array."""
        if self._local_offsets is None:
            self._local_offsets = exscan_with_total(
                    self._local_lengths, dtype=self.size_type)
        return self._local_offsets

    def _local_to_traj_timestep(self, indices: Union[int, Sequence[int]]):
        """Compute and return the trajectory local indices and timestep indices
        for given local state indices."""
        if isinstance(indices, list):
            indices = np.asarray(indices)
        offsets = self._get_local_offsets()
        traj_local_index = np.searchsorted(offsets[1:], indices, side='right')
        timestep_index = indices - offsets[traj_local_index]
        return traj_local_index, timestep_index

    def as_states(self, key: _DatasetKey = ()):
        return _HDF5StatesAccessor(self, _key_to_h5path(key))

    def as_trajectories(self, key: _DatasetKey = ()):
        return _HDF5TrajectoriesAccessor(self, _key_to_h5path(key))

    def _get_local(self, path: str, local_index: int) \
            -> Union[h5py.Group, h5py.Dataset]:
        """Get a trajectory group, given its local index."""
        return self.f[self.traj_fmt.format(self._local_to_global[local_index]) + path]

    def _get_global(self, global_index: int) -> h5py.Group:
        """Get a trajectory group, given its global index."""
        return self.f[self.traj_fmt.format(global_index)]

    def _collectively_update_structure(
            self,
            global_indices: Sequence[int],
            trajectories: Sequence[TensorCollection]):
        """Update global structure. Does not update "local" indexing variables."""
        ops = []
        for global_index, trajectory in zip(global_indices, trajectories):
            group_name = self.traj_fmt.format(global_index) + '/'

            def visitor(keys, array):
                key = group_name + '/'.join(keys)
                dataset = self.f.get(key)
                if not isinstance(dataset, h5py.Dataset) \
                        or dataset.shape != array.shape \
                        or dataset.dtype != array.dtype:
                    # Delete.
                    if dataset is not None:
                        ops.append((dataset.name,))
                    # Create.
                    ops.append((global_index, key, array.shape, array.dtype.name))

            trajectory.named_foreach(visitor)

        if self.comm and self.comm.size > 1:
            ops = sum(self.comm.allgather(ops), [])
        for op in ops:
            if len(op) == 1:    # Delete.
                del self.f[op[0]]
            elif len(op) == 4:  # Create.
                self.f.create_dataset(op[1], shape=op[2], dtype=op[3])
            else:
                raise NotImplementedError(op)  # Unreachable.

    def set_trajectories(self,
                         local_indices: Sequence[int],
                         trajectories: Sequence[TensorCollection]):
        """Update trajectories at given indices. The trajectory length may be
        updated. Collective operation."""
        global_indices = self._local_to_global[local_indices]
        self._local_lengths[local_indices] = [len(tr) for tr in trajectories]
        self._local_offsets = None
        self._store_trajectories(global_indices, trajectories)

    def add_trajectories(self, trajectories: Sequence[TensorCollection]):
        global_indices = self._map_new_trajectories(trajectories)
        self._store_trajectories(global_indices, trajectories)

        if self.comm and self.comm.size > 1:
            # Added trajectories might belong to other ranks, we need to flush
            # before starting to read or to update them.
            self.f.flush()
            self.comm.Barrier()

    def _map_new_trajectories(
            self, trajectories: Sequence[TensorCollection]) -> Sequence[int]:
        """Assign global trajectory indices to new trajectories.

        Updates `_local_lengths` and `_next_global_index`, and resets
        `_local_offsets`. The trajectories themselves are not transferred nor
        stored here.

        Returns:
            list of global indices where to store the given trajectories
        """
        lengths = np.array([len(trajectory) for trajectory in trajectories],
                           dtype=self.size_type)
        n = np.array(len(lengths), dtype=self.size_type)
        if self.comm:
            offset = np.array(0, dtype=self.size_type)
            self.comm.Exscan(n, offset)
            offset += self._next_global_index
        else:
            offset = self._next_global_index
        total = self.comm.allreduce(n) if self.comm else n
        self._next_global_index += total
        self._local_lengths.extend(lengths)
        self._local_offsets = None

        new_global = np.arange(offset, offset + n)
        self._local_to_global.extend(new_global)
        return new_global

    def _store_trajectories(self,
                            global_indices: Sequence[int],
                            trajectories: Sequence[TensorCollection]):
        """Collectively store the trajectories to given global indices.

        Used by `add_trajectories` and `set_trajectories`"""
        trajectories = [t.cpu_numpy() for t in trajectories]
        self._collectively_update_structure(global_indices, trajectories)
        self._update_trajectories(global_indices, (), trajectories)

    def update_states(self, local_indices: Sequence[int], key: Tuple[str],
                      values: Sequence[TensorCollection]) -> None:
        """Update states with given indices. Local operation."""
        assert isinstance(key, tuple), key
        local_traj_indices, timestep_indices = \
                self._local_to_traj_timestep(local_indices)
        global_traj_indices = self._local_to_global[local_traj_indices]
        for global_traj_index, timestep_index, value in \
                zip(global_traj_indices, timestep_indices, values):
            group = self._get_global(global_traj_index)
            if key:
                group = group['/'.join(key)]

            if isinstance(value, TensorCollection):
                def update(keys, array):
                    group['/'.join(keys)][timestep_index] = array

                value.named_foreach(update)
            else:
                group[timestep_index] = value

    def update_trajectories(self, local_indices: Sequence[int], key: Tuple[str],
                            values: Sequence[Any]) -> None:
        """Update trajectories data, without modifying the trajectory length.
        Local operation."""
        assert isinstance(key, tuple), key
        self._update_trajectories(
                self._local_to_global[local_indices], key,
                map(adaled.to_numpy, values))

    def _update_trajectories(self, global_indices: Sequence[int], key: Tuple[str],
                             values: Sequence[Any]) -> None:
        """Update trajectory data without changing the length. Local operation."""
        for global_index, value in zip(global_indices, values):
            group = self._get_global(global_index)
            if key:
                group = group['/'.join(key)]

            if isinstance(value, TensorCollection):
                def update(keys, array):
                    group['/'.join(keys)][:] = array

                value.named_foreach(update)
            else:
                group[:] = value
