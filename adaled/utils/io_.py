from adaled.backends import TensorCollection
from adaled.utils.dataclasses_ import \
        DataclassMixin, PrettyPrintJSONEncoder, SPECIFIED_LATER, dataclass

import numpy as np
import torch

from typing import Any, Callable, Optional
import json
import os
import pickle
import re
import shutil
import warnings

__all__ = [
    'DumpConfig', 'make_symlink', 'recursive_rm', 'reflink_copy',
    'save_executable_script', 'save', 'load', 'load_csv',
]

REMOVE_CURLY_BRACES_RE = re.compile(r'{[^{}]*}')

# For controlling whether torch tensors are loaded to CPU or GPU.
map_location = None
if not torch.cuda.is_available():
    # Automatically import data to CPU memory if CUDA is not available.
    map_location = 'cpu'


@dataclass
class DumpConfig(DataclassMixin):
    """
    Attributes:
        every (int): Frequency of dumping, measured in cycles or steps or
            other, depending on dw the context. In runs with multiple server
            ranks, this refers to the global number of cycles. Adjust the
            frequency to the number of ranks.
        path_fmt (str): Format of the target file name. See Format Attributes.
        latest_symlink_path (str, optional):
            Path of the symlink to the latest dump, or empty string or `None`
            to disable, or `'<auto>'` to automatically infer from `path_fmt`.
        keep_last (int): Number of previous snapshots to keep. -1 to disable. 0 to
            keep only the newest. The old snapshots are deleted after the new
            ones are stored.
        hardlink (bool): Store a hardlink instead of making a copy. Applicable
            to only some object, e.g. HDF5 datasets.

    Format Attributes:
        Depending on the context, different variables will be available for
        formatting:
            `cycle`: current cycle index
            `step`: current step
            `frame`: the index of the dump
    """
    every: int = 0
    path_fmt: Optional[str] = SPECIFIED_LATER
    latest_symlink_path: Optional[str] = '<auto>'
    keep_last: int = 0
    hardlink: bool = False

    def get_latest_symlink_path(self) -> str:
        if self.latest_symlink_path == '<auto>':
            return _auto_symlink_path(self.path_fmt)
        else:
            return self.latest_symlink_path

    def validate(self, prefix: Optional[str] = None):
        if self.keep_last >= 0 \
                and '{frame' not in self.path_fmt \
                and '{cycle' not in self.path_fmt \
                and self.every != 0:  # Don't complain if the dumping is disabled.
            raise ValueError(f"{prefix}: keep_last >= 0 cannot be used with a "
                             f"path_fmt that contains no 'frame' or 'cycle': "
                             f"keep_last={self.keep_last} path_fmt={self.path_fmt!r}")

    def check(self, *,
              cycle: int = None,
              dump_func: Callable[..., None],
              comm=None,
              collective: Optional[bool] = None):
        """Invoke the dump function if the `cycle` is divisible by `every`.

        Warning: if `keep_last` is non-negative, the `keep_last + 1`th previous
        snapshot will be deleted!

        Arguments:
            cycle: (int) current cycle
            dump_func: (callable) function to invoke
            comm: (optional, mpi4py.Intracomm) communicator, adds
            collective: (optional, bool) if False, `rank` keyword will be
                        included in path formatting, must be specified if and
                        only if `comm` is specified.
        """
        if (collective is None) != (comm is None):
            raise TypeError("either both `collective` and `comm` must be "
                            "specified or neither")
        if self.every == 0:
            return

        if cycle is not None:
            if cycle % self.every > 0:
                return
            fmt_kwargs = {'cycle': cycle}
            fmt_kwargs_old = {'cycle': cycle - self.every * (self.keep_last + 1)}
            frame = cycle // self.every
        else:
            # Handle other parametrization here?
            raise TypeError("expected `cycle`")

        if comm is not None and not collective:
            # If rank not given and the format requires it, throw an exception.
            fmt_kwargs['rank'] = comm.rank
            fmt_kwargs_old['rank'] = comm.rank
            if comm.rank >= 1 and '{rank' not in self.path_fmt:
                raise ValueError(
                        f"'{{rank(...)}}' not found in the path format "
                        f"'{self.path_fmt}', multiple ranks potentially "
                        f"writing to the same file")
        elif '{rank' in self.path_fmt:
            raise ValueError(f"unexpected '{rank(...)}' in the path format "
                             f"`{path_fmt}`: comm={comm}, collective={collective}")

        path = self.path_fmt.format(frame=frame, **fmt_kwargs)
        kwargs = {'path': path}
        if self.hardlink:
            kwargs['hardlink'] = True
        dump_func(**kwargs)

        if self.latest_symlink_path and not (collective and comm.rank > 0):
            make_symlink(path, self.get_latest_symlink_path())

        # Assuming 1-based frame.
        if self.keep_last >= 0 and frame > self.keep_last + 1 \
                and (comm is None or comm.rank == 0):
            path_old = self.path_fmt.format(
                    frame=frame - self.keep_last - 1, **fmt_kwargs_old)
            print("Removing old dump: " + path_old)
            recursive_rm(path_old)


def _auto_symlink_path(path_fmt):
    """Return the user-provided path for the 'latest' symlink or construct
    it from the `path_fmt` string.

    Note: the current implementation does not work for nested curly braces!
    """
    if not path_fmt:
        raise TypeError("expected non-empty `path_fmt`")
    path = REMOVE_CURLY_BRACES_RE.sub('', path_fmt)

    # Append '-latest' before the extension for files and before '/' for folders.
    if path.endswith('/'):
        ext = ''
        while path.endswith('/') or path.endswith('\\'):
            # ext += path[-1]  # Symlink should NOT have trailing /.
            path = path[:-1]
    else:
        path, ext = os.path.splitext(path)

    if not path.endswith('-'):
        path += '-'
    return path + 'latest' + ext


def make_symlink(src: str, dst: str, hint_fmt: Optional[str] = None,
                 verbose: bool = True):
    """Create a symlink pointing to `src` named `dst`.

    If `dst` is equal to `'<auto>'`, it is automatically inferred from the
    `hint_fmt` format string, by removing all curly braces and the content
    within, and appending `-latest`.

    This function is a wrapper around `os.symlink` that automatically cleans up
    the old link `dst` and automatically computes the relative path between
    `src` and `dst`.

    If a file at `dst` already exists and is not a symlink, an exception is
    raised.
    """
    if dst == '<auto>':
        dst = _auto_symlink_path(hint_fmt)
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.exists(dst):
        raise RuntimeError(f"destination `{dst}` exists and is not a symlink")
    assert os.path.exists(src)
    is_dir = os.path.isdir(dst)
    if not src.startswith(os.sep):
        src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(src, dst, target_is_directory=is_dir)
    if verbose:
        print(f"Created symlink {dst} --> {src}.")


def recursive_rm(path: str):
    """Emulate `rm -r <path>`, deletes file or folder `path`."""
    try:
        os.remove(path)
        return
    except IsADirectoryError:  # Works on Linux, not on MacOS.
        pass
    except OSError:
        # https://github.com/python/cpython/issues/95815
        if not os.path.isdir(path):
            raise
    shutil.rmtree(path)


def reflink_copy(src, dst, *, quiet: bool = False):
    """Copy file `src` to `dst` using the copy-on-write reflink mechanism, if
    such is available.

    Implementation requires the `reflink` package. Falls back to `shutil.copy2`
    if `reflink` is not available or if reflink copies are not possible for
    given `src` and `dst`.

    If `quiet` is `True`, no warnings will be emitted when `reflink` is
    unavailable of reflink could not be used for some other reason.

    Python may get a native support for reflink copying, see:
    https://bugs.python.org/issue37157
    """
    try:
        import reflink
        reflink.reflink(src, dst)
        return
    except ImportError:
        msg = "`reflink` package not found"
    except NotImplementedError:
        msg = "OS does not support reflinks"
    except reflink.ReflinkImpossibleError:
        msg = "cannot use reflink-copy between the source and destination"

    if not quiet:
        msg += ", falling back to regular copying"
        warnings.warn(msg, stacklevel=2)

    import shutil
    shutil.copy2(src, dst)


def save_executable_script(path: str, content: str, verbose: bool = True):
    """Create an executable script of given content (applies `chmod +x`)."""
    import stat

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if verbose:
        print(f"Saved executable script: {path}")


def save(obj: Any,
         path: str, *,
         symlink: Optional[str] = None,
         verbose: bool = True,
         pickle_protocol: int = 4,
         **kwargs) -> None:
    """Convenience function to save arbitrary objects.

    Arguments:
        obj: (any) object to save
        path: (str) target path
        symlink: (str, optional) if specified, a symlink `symlink` pointing to
                 `path` is created
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith('.h5'):
        from adaled.utils.io_hdf5 import save_hdf5
        save_hdf5(obj, path)
    elif path.endswith('.json'):
        kwargs.setdefault('cls', PrettyPrintJSONEncoder)
        kwargs.setdefault('indent', 4)
        obj = json.dumps(obj, **kwargs)
        with open(path, 'w') as f:
            f.write(obj)
    elif path.endswith('.txt') or path.endswith('.sh'):
        # For executable .sh files, use save_executable_script.
        with open(path, 'w') as f:
            f.write(obj)
    else:
        import torch
        torch.save(obj, path, pickle_protocol=pickle_protocol, **kwargs)

    if symlink:
        os.makedirs(os.path.dirname(os.path.abspath(symlink)), exist_ok=True)
        make_symlink(path, symlink)

    if verbose:
        print(f"Saved {path}")


def load(path: str) -> Any:
    if path.endswith('.csv'):
        return load_csv(path)
    elif path.endswith('.h5'):
        from adaled.utils.io_hdf5 import load_hdf5
        return load_hdf5(path)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        import torch
        return torch.load(path, map_location=map_location)


def load_csv(path: str, delimiter: str = ',') -> TensorCollection:
    """Load a table from a CSV file as a TensorCollection. Assumes the first
    line is comma-separated list of column names."""
    with open(path) as f:
        header = f.readline().strip()
        data = np.loadtxt(f, delimiter=delimiter, ndmin=2)
        data = data.T

        names = header.split(delimiter)
        if len(names) != len(data):
            warnings.warn(f"mismatch between number of columns in the header "
                          f"and the data {len(names)} vs {len(data)}")
            while len(names) < len(data):
                names.append(f'column{len(names)}')

    return TensorCollection(dict(zip(names, data)))
