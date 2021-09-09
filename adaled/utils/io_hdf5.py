from adaled.backends import TensorCollection
import adaled

import h5py
import numpy as np

from typing import Callable, Optional, Tuple, Union

_DatasetLoadFunc = Callable[[h5py.Dataset], Optional[Union[np.ndarray, bool]]]
_GeneralizedSlice = Union[int, slice, Tuple[Union[int, slice], ...]]

class H5PyFile(h5py.File):
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(
                    "NOTE: If h5py is crashing for no apparent reason, check "
                    "if HDF5 is used by another library within the same "
                    "process (e.g. by the micro solver), as there might be "
                    "incompatibilities between different versions of HDF5. "
                    "Try to (a) disable HDF5 in the solver, (b) save in .pt "
                    "instead of .h5, or to (c) perform HDF5 operations on "
                    "other ranks.") from exc_value
        super().__exit__(exc_type, exc_value, traceback)


def load_hdf5_group(group: h5py.Group, dataset_load_func: _DatasetLoadFunc = None):
    """Load selectively data from an HDF5 group.

    Arguments:
        group: (h5py.Group)
        dataset_load_func:
                (callable) function that gets a dataset as input and returns
                either (1) the loaded dataset (or a slice of it) (2) boolean,
                denoting whether to load the dataset or not, (3) `None` to skip
                the dataset
    """
    if dataset_load_func is None:
        dataset_load_func = lambda dataset: dataset[()]
    out = TensorCollection()
    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        data = dataset_load_func(obj)
        if data is None:
            return
        elif isinstance(data, bool):
            if data:
                data = obj[()]
            else:
                return
        elif isinstance(data, h5py.Dataset):
            data = obj[()]
        elif not isinstance(data, (np.ndarray, np.generic)):
            raise TypeError(f"unexpected type: {data.__class__.__mro__}")
        x = out
        *prefix, last = name.split('/')
        for key in prefix:
            if key not in x:
                x[key] = TensorCollection()
            x = x[key]
        x[last] = data

    group.visititems(visitor)
    return out


def load_hdf5(path: str, dataset_load_func: _DatasetLoadFunc = None):
    with H5PyFile(path, 'r') as f:
        return load_hdf5_group(f, dataset_load_func)


def update_hdf5_group(
        group: Union[h5py.Group, h5py.Dataset],
        data: Union[TensorCollection, np.ndarray],
        slice: _GeneralizedSlice = ()):
    """Update existing dataset or group. No datasets are created, deleted nor
    resized.

    In case of an error (i.e. due to mismatched hierarchy or slices), there are
    no guarantees on what fields will be stored or not."""
    if isinstance(group, h5py.Dataset):
        data = adaled.to_numpy(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"to update the dataset {group}, an array "
                            f"has to provided, not {data}")
        group[slice] = data
        return
    if not isinstance(data, (dict, TensorCollection)):
        raise TypeError(f"to update the group {group}, a tensor collection "
                        f"has to be provided, not {data}")

    for key, value in data.items():
        update_hdf5_group(group[key], value, slice)


def save_hdf5(obj: Union[dict, TensorCollection], path: str):
    """Save a TensorCollection into an HDF5 file."""
    if isinstance(obj, dict):
        obj = TensorCollection(obj, default_numpy=True)
    if not isinstance(obj, TensorCollection):
        raise TypeError(f"HDF5 exporting supports only TensorCollections, "
                        f"got {obj.__class__}")

    with H5PyFile(path, 'w') as f:
        def save(keys, x):
            f.create_dataset('/' + '/'.join(keys), data=x)

        obj.cpu_numpy().named_foreach(save)
