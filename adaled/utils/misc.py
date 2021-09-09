from adaled.backends import cmap

import numpy as np

from typing import Any, Callable, Optional, Sequence

def batch_mse(a, b):
    """For each batch item (first dimension), compute the MSE summed over all
    other dimensions (e.g. trajectory and state variables)."""
    dx = a - b
    error = (dx * dx).mean(axis=tuple(range(1, dx.ndim)))
    return error


def exscan_with_total(array: Sequence[Any], dtype=None):
    """Compute an exclusive scan that includes the total sum as an extra
    element.

    >>> exscan_with_total(np.array([10, 20, 30, 40]))
    [0, 10, 30, 60, 100]
    """
    out = np.cumsum(array, dtype=dtype)
    out = np.insert(out, 0, 0)
    return out


def function_to_generator(
        func: Callable[[float], Any],
        dt: float,
        t: float = 0.0):
    """Convert a continuous-time function to an infinite generator that
    evaluates the function at t, t+dt, t+2*dt etc.

    Arguments:
        func: function to evaluate
        dt: time step
        t: initial time
    """
    while True:
        yield func(t)
        t += dt


_global_np_dtype = np.float32  # torch's default is float32

def get_global_default_np_dtype():
    return _global_np_dtype


def set_global_default_dtype(dtype):
    global _global_np_dtype
    import torch
    if dtype == np.float32 or dtype == torch.float32:
        _global_np_dtype = np.float32
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
    elif dtype == np.float64 or dtype == torch.float64:
        _global_np_dtype = np.float64
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"expected float32 or float64, got {np_dtype}")


def init_torch(
        server_comm: Optional['mpi4py.MPI.Intracomm'] = None,
        cuda: bool = True,
        group_size: int = 1,
        double_precision: bool = True):
    """Initialize torch and horovod if world size larger than 1.

    NOTE: It is very important to pass the server-only communicator and not the
    whole world communicator. As soon as the server_comm has more than 1 rank,
    horovod is initialized, which adds a busy-wait background thread. If this
    thread is added for clients it might slow down the execution of the
    simulation!
    """
    import torch
    if server_comm and server_comm.size > 1:
        from mpi4py import MPI
        import horovod.torch  # Required.
        horovod.torch.init(server_comm)
        if torch.cuda.is_available():
            local_rank = server_comm.Split_type(MPI.COMM_TYPE_SHARED).rank
            group_id = local_rank // group_size
            if torch.cuda.device_count() > 1:
                raise NotImplementedError(
                        "Handling multiple devices not implemented, there "
                        "seems to be a problem with pickling and sending data "
                        "between ranks. Instead, add a wrapper scripts that "
                        "sets CUDA_VISIBLE_DEVICES for each rank separately.")
            gpu_id = group_id % torch.cuda.device_count()
            print(f"Rank: {local_rank}  group_id: {group_id}  gpu_id: {gpu_id}")
            torch.cuda.set_device(gpu_id)

    set_global_default_dtype(np.float64 if double_precision else np.float32)


def _merge(a):
    shape = a.shape
    return a.reshape(shape[0] * shape[1], *shape[2:])


def merge_axes_01(x):
    """Merge axes 0 and 1 of a tensor or a tensor collection."""
    return cmap(_merge, x)


def split_first_axis(x, shape):
    """Reshape the first axis to match the given shape."""
    def _split(a):
        return a.reshape(*shape, *a.shape[1:])

    return cmap(_split, x)


def to_numpy(x):
    # FIXME: Is this really necessary, why not simply cmap(np.asarray, ...)?
    if hasattr(x, 'numpy'):
        return x.cpu().numpy()
    if isinstance(x, list):
        raise TypeError(f"lists should be handled manually")
    if not isinstance(x, (np.ndarray, np.generic)):
        try:
            x = x[:]  # DynamicArray, datasets...
        except:
            pass
        else:
            return to_numpy(x)
        raise TypeError(f"unrecognized type `{x.__class__}`: {x}")
    return x


def to_numpy_nonstrict(x):
    if isinstance(x, list):
        return np.asarray(x)
    return to_numpy(x)


def set_highlighted_excepthook():
    """Colorize exceptions using `pygments` package.

    Silently fails if `pygments` not found.
    """
    # https://stackoverflow.com/questions/14775916/coloring-exceptions-from-python-on-a-terminal
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import TerminalFormatter
    except ImportError:
        return
    import sys, traceback

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
    formatter = TerminalFormatter()

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook
