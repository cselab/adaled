from adaled import AdaLEDStage, DynamicArray, TensorCollection
from adaled.utils.dataclasses_ import dataclass, field
import adaled

import numpy as np

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import functools
import glob
import math
import os
import re

_MultikeyFilterFunc = Callable[[Tuple[str, ...]], Optional[bool]]

__all__ = ['load_record']


@dataclass
class CycleList:
    """A collection of cycles and some of the subsets."""
    whole: List[TensorCollection] = field(list)
    last_cmp_to_last_macro: List[TensorCollection] = field(list)


def _get_fields_and_cycles_slices(record):
    fields = record['fields']
    try:
        len(fields)
    except:
        raise Exception("all fields must be recorded at the same frequency "
                        "(check `every`, `x_every` and `z_every` attributes "
                        "of the `RecorderConfig`)")
    slices = get_cycle_slices(fields['metadata', 'stage'])
    return fields, slices


def compute_macro_utilization(t: np.ndarray, stages: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Compute the start times and macro utilizations of every cycle.

    Arguments:
        t: array of time step indices
        stages: array of stages, one element per time step

    Outputs:
        cycle_t: array of cycle starts
        cycle_utilization: array of cycle macro utilizations
    """
    is_macro = stages == int(AdaLEDStage.MACRO)
    slices = get_cycle_slices(stages)
    cycle_t = np.empty(len(slices) + 1)
    cycle_utilization = np.empty(len(slices))
    for i, (begin, end) in enumerate(slices):
        cycle_t[i] = t[begin]
        cycle_utilization[i] = is_macro[begin:end].sum() / (end - begin)
    cycle_t[-1] = end

    return cycle_t, cycle_utilization


def filter_and_slice_accepted_cycles(record: TensorCollection) -> CycleList:
    """Filter cycles where macro solver was accepted. Return only the part
    starting from the last comparison step, ending in the last macro step, both
    inclusive."""
    fields, slices = _get_fields_and_cycles_slices(record)

    out = CycleList()
    for begin, end in slices:
        cycle: TensorCollection = fields[begin:end]
        stage = cycle['metadata', 'stage']

        last_macro = (stage == AdaLEDStage.MACRO).nonzero()[0]
        if len(last_macro) == 0:
            continue  # Macro was not accepted.
        last_cmp = (stage == AdaLEDStage.COMPARISON).nonzero()[0][-1]
        last_macro = last_macro[-1]

        out.whole.append(cycle)

        part = cycle[last_cmp:last_macro + 1]
        stage = part['metadata', 'stage']
        assert stage[0] == AdaLEDStage.COMPARISON
        assert (stage[1:] == AdaLEDStage.MACRO).all()
        out.last_cmp_to_last_macro.append(part)

    return out


def get_rejected_cycles_last_cmp_step(record: TensorCollection) \
        -> List[TensorCollection]:
    """Return the list of last comparison steps of all rejected cycles."""
    fields, slices = _get_fields_and_cycles_slices(record)

    last_cmps = DynamicArray()
    for begin, end in slices:
        cycle: TensorCollection = fields[begin:end]
        stage = cycle['metadata', 'stage']

        last_macro = (stage == AdaLEDStage.MACRO).nonzero()[0]
        if len(last_macro) > 0:
            continue  # Macro was accepted.

        last_cmp = (stage == AdaLEDStage.COMPARISON).nonzero()[0][-1]
        last_cmp = cycle[last_cmp]
        assert last_cmp['metadata', 'stage'] == AdaLEDStage.COMPARISON
        last_cmps.append(last_cmp)

    return last_cmps.data


def get_cycle_slices(stages: np.ndarray) -> np.ndarray:
    """Return a Cx2 array of beginning (inclusive) and ending (exclusive) of
    each cycle, where C is the number of cycles.

    This function assumes that all timesteps are recorded. If some time steps
    are skipped, the result might be inaccurate.
    """
    Stage = adaled.AdaLEDStage
    is_start = (stages == Stage.WARMUP) | (stages == Stage.COMPARISON)
    is_start = np.concatenate([[False], is_start, [True]])
    is_start = ~is_start[:-1] & is_start[1:]
    idx = is_start.nonzero()[0]
    return np.stack([idx[:-1], idx[1:]], axis=1)


def normalize_record(record: TensorCollection, version: Optional[int] = None):
    """Normalize record format to the latest version."""
    if version is None:
        version = record['version']
    elif 'version' not in record:
        record = TensorCollection(**record.dict, version=version)
    elif isinstance(record, dict):  # Backward compatibility, 2022-02-04.
        record = TensorCollection(record)

    # FIXME: Remove this, fix the cubism setup instead.
    try:
        if record['fields', 'simulations', 'validation'].shape == (0,):
            print(f"HOTFIX: removing empty fields/simulations/validation.")
            del record['fields', 'simulations']['validation']
    except KeyError:
        pass

    return record


def slice_record_fields(fields: TensorCollection, s: slice):
    """Slice with respect to time."""
    ref = fields['metadata']  # Reference field.
    start, stop, step = s.indices(len(ref))
    if step != 1:
        raise NotImplementedError("step != 1 not supported")
    ratios = [len(field) // len(ref) for field in fields.allvalues()]
    gcd = functools.reduce(math.gcd, ratios)
    if start % gcd != 0 or stop % gcd != 0:
        raise ValueError(f"start={start} and stop={stop} (s={s}) must be "
                         f"multiples of {gcd} (see x_every and z_every)")

    def take_slice(field):
        ratio = len(ref) // len(field)
        print(field.shape, start, stop, len(field), len(ref))
        print(field.shape, start, stop, ratio, start // ratio, stop // ratio)
        return field[start // ratio:stop // ratio]

    return fields.map(take_slice)


def slice_record(record: TensorCollection, s: slice):
    out = record.map(lambda x: x)  # Shallow copy.
    out['fields'] = slice_record_fields(out['fields'], s)
    return out


def slice_record_trajectory_batch(
        record: TensorCollection,
        s: Union[int, slice]) -> TensorCollection:
    """Slice with respect to batch."""
    record = record.map(lambda x: x)  # Shallow copy.
    record['fields']['simulations'] = record['fields']['simulations'][:, s]
    return record


def load_record(
        path_patterns: Optional[Union[str, Sequence[str]]] = None,
        filter_func: Optional[Callable[[Tuple[str, ...], 'h5py.Dataset'], Optional[Any]]] = None,
        *,
        regex: Union[str, re.Pattern] = None,
        max_frames: int = -1):
    """Load and concatenate record files.

    Arguments:
        path_patterns: (none or str or list of strs)
                Record file paths glob patterns. If None, load_record will
                automatically look for `record-*.h5/pt` files, excluding
                `record-*latest*.*` files. Automatic lookup may be unsuitable
                for multi-client runs.
        max_frames: (int, optional) maximum number of frames to load
        filter_func: (callable, optional) callable that takes the tensor
                     multikey (tuple) and the tensor and returns either the
                     slice of the tensor or None to skip this value (*)
        regex: if set, load (whole) datasets whose name matches the given regex

    (*) For HDF5 files, the passed tensor is an `h5py.Dataset`. Returning
    `None` prevents loading this dataset altogether. Or, if a slice is taken,
    only that slice will be read.
    """
    if (regex is not None) and (filter_func is not None):
        raise TypeError("cannot specify both regex and filter_func")

    from adaled.utils.glob_ import glob_ex

    if path_patterns is None:
        paths = glob_ex(include=['record-*.h5', 'record-*.pt'],
                        exclude=['record-*latest*.*'])
        if not paths:
            raise ValueError("automatic lookup of record files failed, specify"
                             " record file paths or glob pattern manually")
    else:
        paths = glob_ex(path_patterns)

    if regex:
        regex = re.compile(regex)

    def h5_filter_func(d):
        if d.name == '/version':  # Always check version.
            nonlocal version
            version = d[()]
        if filter_func:
            return filter_func(tuple(d.name.split('/'))[1:], d)
        elif regex:
            return bool(regex.match(d.name))
        else:
            return d[()]

    fields = defaultdict(adaled.DynamicArray)
    version = None
    for i, path in enumerate(paths):
        print("Loading", path)
        if path.endswith('.pt'):
            record = normalize_record(adaled.load(path))
            version = record.get('version')
            if filter_func:
                record = record.named_map(filter_func)
        elif path.endswith('.h5'):
            # Ideally we would like to normalize the record before invoking
            # `filter_func`. Since normalization is about deprecated, it
            # doesn't really matter.
            from adaled.utils.io_hdf5 import load_hdf5  # Lazy load.
            record = load_hdf5(path, h5_filter_func)
            record = normalize_record(record, version)
        else:
            raise ValueError("unrecognized format: " + path)
        record = record.remove_empty()
        for key, value in record['fields'].items():
            fields[key].extend(value)
        if max_frames >= 0 and \
                'simulations' in fields.keys() \
                and 'F' in fields['simulations'].data.keys() \
                and len(fields['simulations']['F']) >= max_frames:
            fields = slice_record_fields(TensorCollection(fields), slice(0, max_frames))
            break

    # Convert to proper arrays if still DynamicArrays.
    fields = {k: v[:] for k, v in fields.items()}

    if version is None:
        return TensorCollection(fields=fields)
    else:
        return TensorCollection(version=version, fields=fields)


# TODO: Remove the old name.
load_and_concat_records = load_record


def _make_h5_small_data_filter(
        version: int,
        trajectory_slice: Union[int, slice],
        to_load_regex: Optional[re.compile]):
    def only_small_filter(k: Tuple[str, ...], d: 'h5py.Dataset'):
        if to_load_regex and to_load_regex.match(d.name):
            return d[:, trajectory_slice]

        if k[0] == 'fields':
            if k[1] == 'metadata':
                return d
            else:
                return None
        elif k[0] == 'version':
            return True
        else:
            return None  # Skip the rest for now.

    return only_small_filter


def _make_h5_x_filter(
        version: int,
        frame: int,
        trajectory_slice: Union[int, slice],
        frames_per_record: int,
        filter_func: Optional[_MultikeyFilterFunc]):
    def only_x_filter(d: 'h5py.Dataset'):
        multikey = tuple(d.name.split('/')[1:])
        should_load = None
        if filter_func:
            should_load = filter_func(multikey)
        if should_load is None:
            should_load = (
                    multikey[:3] == ('fields', 'simulations', 'x')
                    # v5, legacy 2022-22-04
                    or multikey[:3] == ('data', 'simulations', 'x')
                    # v3, legacy 2021-12-21
                    or multikey[:2] == ('data', 'x'))
        if should_load:
            assert frames_per_record % len(d) == 0, (d, frames_per_record)
            # x data may be stored rarer than metadata.
            return d[frame // (frames_per_record // len(d)),
                     trajectory_slice]

    return only_x_filter


# Note: This could be refactored to behave as a dataset and then simply moved
# to adaled/utils/data/hdf5.py
class LazyRecordLoader:
    """Provides lazy loading of the x (real) part of the dataset.

    Preloads latent space states and other small data.
    """
    def __init__(self,
                 paths: Sequence[str],
                 max_frames: int = -1,
                 trajectory_slice: Union[slice, int] = 0,
                 load_immediately_regex: Optional[str] = r'/?fields/simulations/(F|uncertainty|z)'):
        """
        Arguments:
            paths: (list of strings) records to load
            max_frames: (int) maximum number of frames to load, -1 for no limit
            trajectory_slice: (slice or int) which parallel trajectories to
                              load, defaults to 0 (only the first trajectory)
            load_immediately: keys of simulation data to load immediately,
                              available as `self.small_fields`
        """
        from adaled.utils.io_hdf5 import load_hdf5
        if not paths:
            raise ValueError("expected at least one path")

        load_immediately_regex = re.compile(load_immediately_regex) \
                if load_immediately_regex else None

        def load_first(d):
            if d.name == '/version':
                self.version = d[()]
            elif d.name == '/fields/metadata/stage':
                self._frames_per_record = len(d)
            elif d.name == '/data/stage':  # v5, legacy since 2022-02-10.
                self._frames_per_record = len(d)
            return None

        self.version: Optional[int] = None
        self._frames_per_record: Optional[int] = None
        load_hdf5(paths[0], load_first)
        assert self.version is not None
        assert self._frames_per_record is not None

        filter = _make_h5_small_data_filter(
                self.version, trajectory_slice, to_load_regex=load_immediately_regex)
        self.small_fields = load_record(
                paths, filter, max_frames=max_frames)['fields']
        self._paths = paths

    def get_frame_x(
            self,
            i: int,
            trajectory_slice: Union[slice, int] = slice(None),
            filter_func: Optional[_MultikeyFilterFunc] = None):
        """Load and return the x-data of the i-th frame.

        If `x` is stored at a lower frequency than the metadata, the first
        previous frame is loaded.

        Arguments:
            trajectory_slice: (slice or int) which parallel trajectory to load
            filter_func: (optional) A function that takes a field multikey
                    (tuple of strings) as input and returns whether the field
                    should be loaded or not. If the function returns `None`,
                    the default criterion is used.
        """
        from adaled.utils.io_hdf5 import load_hdf5
        if i >= len(self.small_fields):
            raise IndexError(f"requested time step #{i} when only "
                             f"{len(self.small_fields)} time steps were recorded")
        path = self._paths[i // self._frames_per_record]
        frame = i % self._frames_per_record

        filter = _make_h5_x_filter(
                self.version, frame, trajectory_slice, self._frames_per_record,
                filter_func)
        record = normalize_record(load_hdf5(path, filter), version=self.version)
        fields = record['fields']
        sim = fields.get('simulations', fields)  # v3 had no 'simulations' (2021-12-21)
        return sim
