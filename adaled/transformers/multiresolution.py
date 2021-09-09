from adaled.backends import Backend, get_backend
from adaled.utils.dataclasses_ import DataclassMixin, dataclass, field

import torch.nn.functional as F
import torch
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math

_Array = Union[np.ndarray, torch.Tensor]
_FCoord = Tuple[float, ...]
_ICoord = Tuple[int, ...]
_IBox = Tuple[_ICoord, _ICoord]
_float_like = (float, np.floating)
_int_like = (int, np.integer)

def sigmoid(x):
    with np.errstate(over='ignore'):
        return 1 / (1 + np.exp(-x))


# Parameters for torch.nn.functional.interpolate, used for upscaling.
DEFAULT_UPSCALING_KWARGS = {
    'mode': 'AUTO_LINEAR',  # linear, bilinear or trilinear
    'align_corners': False,
}
_MODE_LINEAR = ['linear', 'bilinear', 'trilinear']

def _relative_slice(outer_box: _IBox, inner_box: _IBox) -> Tuple[slice]:
    out = []
    for outer0, outer1, inner0, inner1 in zip(*outer_box, *inner_box):
        assert outer0 <= outer1
        assert inner0 <= inner1
        if not (outer0 <= inner0 <= inner1 <= outer1):
            raise ValueError(f"only nested hierarchy supported: "
                             f"outer={outer_box} inner={inner_box}")
        shift = inner0 - outer0
        out.append(slice(shift, shift + inner1 - inner0))
    return (...,) + tuple(out[::-1])


def _to_int_coords(coord: Union[_FCoord, _ICoord], grid: _ICoord) -> _ICoord:
    scale = max(grid)
    if len(coord) != len(grid):
        raise TypeError(f"expected length {len(grid)}, got {coord}")

    if all(isinstance(c, _int_like) for c in coord):
        return tuple(coord)
    elif all(isinstance(c, _float_like) for c in coord):
        return tuple(int(round(c * scale)) for c in coord)
    else:
        raise TypeError(f"expected either all ints or all floats, got {coord}")


def _to_tuple(x: Union[int, _ICoord], ndim: int) -> _ICoord:
    if isinstance(x, _int_like):
        return (x,) * ndim
    else:
        if len(x) != ndim:
            raise TypeError(f"expected a tuple of length {ndim}, got {x}")
        return tuple(x)


def _expand_to_round(
        begin: _ICoord,
        end: _ICoord,
        round_size_to: _ICoord,
        outer_begin: _ICoord,
        outer_end: _ICoord) -> Tuple[_ICoord, _ICoord]:
    assert len(begin) == len(end) == len(outer_begin) == len(outer_end)
    out_begin = []
    out_end = []
    for d, (b, e, r, ob, oe) in \
            enumerate(zip(begin, end, round_size_to, outer_begin, outer_end)):
        if not (ob <= b <= e <= oe):
            raise ValueError(
                    f"hierarchy of layers not nested: inner={begin}--{end} "
                    f"outer={outer_begin}--{outer_end}")
        left = (r - (e - b) % r) % r
        while left > 0:
            if ob < b:
                b -= 1
                left -= 1
                if left > 0 and e < oe:
                    e += 1
                    left -= 1
            elif e < oe:
                e += 1
                left -= 1
            else:
                raise ValueError(
                        f"cannot ensure that the layer size in dimension "
                        f"{d} is divisible by {r}:  inner={begin}--{end} "
                        f"outer={outer_begin}--{outer_end}")
        out_begin.append(b)
        out_end.append(e)

    out_begin = tuple(out_begin)
    out_end = tuple(out_end)
    return (out_begin, out_end)


def _identity_avg_func(array: _Array, out: Optional[_Array]) -> _Array:
    """Identity averaging (downscaling) function. If `out` is given, copies the
    data, otherwise returns the input by reference."""
    if out is None:
        return array  # Returning by reference!
    else:
        out[:] = array
        return out


def _wrap_avg_pool(stride, avg_pool):
    """Decorate an AvgPool layer with optional numpy->torch->numpy conversion
    and optional appending and removing extra dimensions."""
    D = len(stride)
    def _wrapped_avg_pool(array: _Array, out: Optional[_Array]) -> _Array:
        prefix = array.shape[:-D]
        is_numpy = isinstance(array, np.ndarray)
        if is_numpy:
            array = torch.from_numpy(array)
        # AvgPool supports input shapes (N, C, Z, Y, X) or (C, Z, Y, X) only.
        # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
        if not prefix:
            array = array[np.newaxis]
        elif not (1 <= len(prefix) <= 2):
            raise NotImplementedError(
                    f"only 1 or 2 extra dimensions supported, got shape {array.shape}")

        array = avg_pool(array)

        if not prefix:
            array = array[0]
        if is_numpy:
            array = array.numpy()

        if out is None:
            return array
        else:
            # AvgPool* don't seem to support the out argument.
            out[:] = array
            return out

    return _wrapped_avg_pool


@dataclass
class MRLayerConfig(DataclassMixin):
    center: Optional[Union[_FCoord, _ICoord]] = None
    size: Optional[Union[_FCoord, _ICoord]] = None
    stride: Union[int, _ICoord] = 1
    round_size_to: Optional[Union[int, _ICoord]] = None

    upscaling_kwargs: Dict[str, Any] = field(lambda: DEFAULT_UPSCALING_KWARGS)

    alpha_margin_cells: float = 10.0
    alpha_sigma_cells: float = 2.0

    def finalize(self, grid: _ICoord, outer_box: Tuple[_ICoord, _ICoord]) \
            -> 'FinalizedLayer':
        grid = tuple(grid)
        D = len(grid)

        if self.center is None \
                and self.size is None \
                and (self.round_size_to is None
                        or all(g % self.round_size_to == 0 for g in grid)):
            return FinalizedLayer(
                    grid, (0,) * D, grid, _to_tuple(self.stride, D),
                    self.upscaling_kwargs, 0.0, 0.0)
        elif self.center is None or self.size is None:
            raise TypeError(f"incompatible center={self.center}, "
                            f"size={self.size}, round_size_to={self.round_size_to} and grid={grid}")

        center = _to_int_coords(self.center, grid)
        size = _to_int_coords(self.size, grid)

        begin = tuple(c - s // 2 for c, s in zip(center, size))
        end = tuple(b + s for b, s in zip(begin, size))

        stride = _to_tuple(self.stride, D) \
                if self.stride is not None else (1,) * D
        round_size_to = _to_tuple(self.round_size_to, D) \
                if self.round_size_to is not None else stride
        if not all(r % s == 0 for r, s in zip(round_size_to, stride)):
            raise ValueError(f"round_size_to={round_size_to} must be divisible "
                             f"by stride={stride}")

        begin, end = _expand_to_round(begin, end, round_size_to, *outer_box)

        return FinalizedLayer(grid, begin, end, stride, self.upscaling_kwargs,
                               self.alpha_margin_cells, self.alpha_sigma_cells)


class FinalizedLayer:
    def __init__(self,
                 grid: _ICoord,
                 begin: _ICoord,
                 end: _ICoord,
                 stride: _ICoord,
                 upscaling_kwargs: Dict[str, Any],
                 alpha_margin_cells: float,
                 alpha_sigma_cells: float):
        assert len(grid) == len(begin) == len(end) == len(stride), \
               (len(grid) == len(begin), len(end), len(stride))
        assert all((e - b) % s == 0 for b, e, s in zip(begin, end, stride)), \
               (begin, end, stride)

        self.grid = grid
        self.ndim = len(begin)

        self.begin = begin
        self.end = end
        self.stride = stride

        self.size = tuple(e - b for b, e in zip(begin, end))
        self.box = (begin, end)
        self.slices = (...,) + tuple(slice(b, e) for b, e in zip(begin, end))[::-1]

        self.upscaling_kwargs = upscaling_kwargs.copy()
        if self.upscaling_kwargs.get('mode') == 'AUTO_LINEAR':
            if 1 <= self.ndim <= 3:
                self.upscaling_kwargs['mode'] = _MODE_LINEAR[self.ndim - 1]
            else:
                raise NotImplementedError("what upscaling mode to use?")

        self.alpha_margin_cells = alpha_margin_cells
        self.alpha_sigma_cells = alpha_sigma_cells

        self.downscaled_shape = \
                tuple(s // st for s, st in zip(self.size, stride))[::-1]
        self.upscaled_shape = self.size[::-1]
        self.downscaled_elements_per_channel = math.prod(self.downscaled_shape)

        self._downscale_func = self.make_downscale_func(stride)
        self._needs_scaling = any(s != 1 for s in stride)

        self._averaging_slice = \
                (..., ) + tuple(slice(b, e, 1) for b, e in zip(begin, end))[::-1]

    def __repr__(self):
        return f"{self.__class__.__name__}(grid={self.grid}, " \
               f"begin={self.begin}, end={self.end}, stride={self.stride}, " \
               f"alpha_margin_cells={self.alpha_margin_cells} " \
               f"alpha_sigma_cells={self.alpha_sigma_cells} " \
               f"upscaling_kwargs={self.upscaling_kwargs!r})"

    def filter_dimensions(self, mask: Sequence[bool]) -> 'FinalizedLayer':
        """Return a new FinalizedLayer with only the dimensions d for which
        mask[d] is True."""
        def _slice(array: _ICoord) -> _ICoord:
            assert len(array) == len(mask), (array, mask)
            return tuple(a for m, a in zip(mask, array) if m)

        upscaling_kwargs = self.upscaling_kwargs.copy()
        if upscaling_kwargs.get('mode') in _MODE_LINEAR:
            upscaling_kwargs['mode'] ='AUTO_LINEAR'

        return self.__class__(
                grid=_slice(self.grid),
                begin=_slice(self.begin),
                end=_slice(self.end),
                stride=_slice(self.stride),
                upscaling_kwargs=upscaling_kwargs,
                alpha_margin_cells=self.alpha_margin_cells,
                alpha_sigma_cells=self.alpha_sigma_cells)

    def make_downscale_func(self, stride: _ICoord) \
            -> Callable[[_Array, Optional[_Array]], _Array]:
        """Create the downscaling function.

        If all strides are 1, an identity function is returned.
        Otherwise, torch.nn.AvgPool1d/2d/3d is used.

        Benchmarks show that a naive `.reshape(...).mean(...)` trick is very
        slow and unparallelized by numpy, while `torch.nn.AvgPool*` functions
        is parallelized and very efficient.
        """
        if any(s != s for s in stride):
            return _identity_avg_func

        if self.ndim == 1:
            layer = torch.nn.AvgPool1d(kernel_size=stride)
        elif self.ndim == 2:
            layer = torch.nn.AvgPool2d(kernel_size=stride)
        elif self.ndim == 3:
            layer = torch.nn.AvgPool3d(kernel_size=stride)
        else:
            raise NotImplementedError("more than 3D not supported")

        return _wrap_avg_pool(stride, layer)

    def slice_and_downscale(self, array: _Array, out: Optional[_Array] = None):
        """Group the last `len(self.stride)` dimensions of an array into blocks
        of shape `self.stride` axes and compute per-block averages.

        Note: for s

        For example:
            >>> layer = FinalizedLayer(grid, stride=(2, 2, 2), begin=(5, 5, 5), end=(15, 17, 19))
            >>> layer.slice_and_downscale(np.ones((2, 3, 30, 30, 30))).shape
            (2, 3, 5, 6, 7)
        """
        if array.shape[-self.ndim:] != self.grid[::-1]:
            raise TypeError(f"expected shape ending in {self.grid[::-1]}, got {array.shape}")
        array = array[self._averaging_slice]
        return self.downscale_sliced(array, out)

    def downscale_sliced(self, array: _Array, out: Optional[_Array] = None):
        D = self.ndim
        if array.shape[-D:] != self.upscaled_shape:
            raise TypeError(f"expected shape ending in {self.upscaled_shape},"
                            f" got {array.shape}")
        return self._downscale_func(array, out)

    def upscale(self, array: _Array, out: Optional[_Array] = None):
        """Upscale the image using torch.nn.functional.interpolate.

        The default settings are `mode='bilinear'` for 2D, `mode='trilinear'`
        for 3D, and `align_corners=False`, which is cell-centered interpolation
        everywhere except for the first and last item (?).

        >>> x = torch.tensor([[[0., 1., 2., 3.]]])
	>>> interpolate(x, scale_factor=2, mode='linear', align_corners=False)
	tensor([[[0.0000, 0.2500, 0.7500, 1.2500, 1.7500, 2.2500, 2.7500, 3.0000]]])
	"""
        D = self.ndim
        if array.shape[-D:] != self.downscaled_shape:
            raise TypeError(f"expected shape {self.downscaled_shape}, got {array.shape}")
        if not self._needs_scaling:
            if out is None:
                return array  # Returning by reference!
            else:
                out[:] = array
                return out

        if out is None:
            out = get_backend(array).empty_like(
                    array, shape=array.shape[:-D] + self.upscaled_shape)
        else:
            if out.shape[-D:] != self.upscaled_shape:
                raise TypeError(f"expected shape ending in {self.upscaled_shape}, "
                                f"got {array.shape}")

        # Ensure shape is (batch, channels, [[z,], y,] x).
        final_out = out
        while out.ndim < 2 + D:
            out = out[None]
        while array.ndim < 2 + D:
            array = array[None]

        # If pytorch complains about non-writeable numpy arrays, try to avoid
        # using operations that make the arrays non-writeable (like
        # np.broadcast_to does).
        if isinstance(array, np.ndarray):
            # as_tensor moves data to the default device, contrary to the docs.
            # https://github.com/pytorch/pytorch/pull/45500
            array = torch.from_numpy(array)
        else:
            array = torch.as_tensor(array)

        for ij in np.ndindex(*array.shape[:-D - 2]):
            out[ij] = F.interpolate(array[ij], scale_factor=self.stride,
                                    **self.upscaling_kwargs)

        # Return original shape, without D+2 padding for torch.
        return final_out

    # TODO: Rename to rectangular function or to sigma?
    def compute_alpha(self, box: _IBox, *, margin: Optional[float] = None, dtype=np.float32):
        """Compute alpha within the given box of cells."""
        xs = [np.arange(size) for size in self.grid]
        alpha = dtype(1.0)
        if margin is None:
            margin = dtype(self.alpha_margin_cells)
        else:
            margin = dtype(margin)

        for d in range(self.ndim):
            # Assumed that cell data is cell-centered (hence 0.5), but that is
            # not really that important here.
            x = np.arange(box[0][d] + 0.5, box[1][d] + 0.5, dtype=dtype)
            dist = np.minimum(x - (self.begin[d] + margin), self.end[d] - margin - x)
            dist /= dtype(self.alpha_sigma_cells)
            factor = sigmoid(dist)
            # Outer product.
            alpha = factor[(...,) + (np.newaxis,) * d] * alpha
        assert alpha.ndim == self.ndim, (alpha.ndim, self.ndim)
        return alpha

    def compute_downscaled_cell_centers(self) -> np.ndarray:
        """Returns an array of points corresponding to the cell centers.

        For 2D, the return shape is (Y, X, 2),
        For 3D, the return shape is (Z, X, Y, 3), and so on.

        The coordinate system is in the cells of the upscaled image.
        """
        slices = [
            slice(begin + 0.5, end - stride + 0.5, (end - begin) // stride * 1j)
            for begin, end, stride in zip(self.begin, self.end, self.stride)
        ]
        out = np.mgrid[slices].T
        assert out.shape == self.downscaled_shape + (self.ndim,), out.shape
        return out


class Multiresolution:
    def __init__(self,
                 layers: Sequence[Union[dict, MRLayerConfig]],
                 grid: _ICoord,
                 alpha_dtype: type = np.float32,
                 dim_mask: Optional[Sequence[bool]] = None):
        """
        Arguments:
            dim_mask: (optional list of bools) if set, keep only the dimensions
                    `d` for which `mask[d]` is `True`
        """
        if dim_mask is not None:
            if len(dim_mask) != len(grid):
                raise ValueError("dim_mask and grid must be of same length")

        finalized_layers = []
        _outer_box = ((0,) * len(grid), grid)
        for layer in layers:
            if isinstance(layer, dict):
                layer = MRLayerConfig(**layer)
            layer.validate()
            layer = layer.finalize(grid, _outer_box)
            finalized_layers.append(layer)
            _outer_box = (layer.begin, layer.end)

        layers = finalized_layers

        if dim_mask is not None:
            # dim_mask must be applied *after* the layers have been built,
            # because the scale is determined by max(grid).
            grid = tuple(g for mask, g in zip(dim_mask, grid) if mask)
            layers = [layer.filter_dimensions(dim_mask) for layer in layers]

        self.layers = layers

        # 0th layer does not need alpha (alpha_0 == 1.0).
        self._alphas = None
        self._alpha_dtype = alpha_dtype
        self._relative_slices = [
            _relative_slice(layers[i].box, layers[i + 1].box)
            for i in range(len(layers) - 1)
        ]
        self.total_elements_per_channel = \
                sum(l.downscaled_elements_per_channel for l in self.layers)
        self.ndim = len(grid)
        self.grid = grid

    def describe(self):
        print("Multiresolution:")
        for i, layer in enumerate(self.layers):
            print(f"    #{i}: {layer}")

    def _compute_alphas(self):
        if self._alphas is None:
            self._alphas = [l.compute_alpha(l.box, dtype=self._alpha_dtype)
                            for l in self.layers[1:]]

    def compute_reconstruction_contributions(self, dtype=np.float32) \
            -> List[np.ndarray]:
        """Compute weight arrays showing reconstruction contribution of each layer."""
        if len(self.layers) == 1:
            return np.ones(self.grid[::-1], dtype=dtype)

        # Construct in reverse order.
        out = []
        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i];
            alpha = np.zeros(self.grid[::-1], dtype=dtype)
            alpha[layer.slices] = layer.compute_alpha(layer.box, dtype=dtype)
            if out:
                alpha -= out[-1]
            out.append(alpha)

        out.append(1 - out[-1])
        return out[::-1]

    def slice_and_downscale(self, array: _Array) -> List[_Array]:
        """Slice and downscale given array, return the list of downscaled
        arrays, one for each layer."""
        return [l.slice_and_downscale(array) for l in self.layers]

    def slice_and_downscale_contiguous(
            self,
            array: _Array,
            out: Optional[_Array] = None) -> Tuple[_Array, List[_Array]]:
        """Slice and downscale the given array and store the output as a single
        contiguous array.

        Returns a pair:
            (flattened concatenated output, list of individual downscaled arrays).
        """
        raise NotImplementedError("not tested")
        prefix = array.shape[:-len(self.layers)]
        num_channels = math.prod(prefix)
        total_size = num_channels * self.total_elements_per_channel
        if out is None:
            out = get_backend(array).empty_like(array, shape=(total_size,))
        elif out.shape != (total_size,):
            raise TypeError(f"expected shape ({total_size},), got {out.shape}")

        out_slices = []
        offset = 0
        for size, layer in zip(sizes, self.layers):
            out_slices.append(
                    layer.slice_and_downscale(state, out=out[offset : offset + size]))
            offset += size
        return out, out_slices

    def rebuild_unsafe(self, arrays: Sequence[np.ndarray], out: Optional[np.ndarray] = None):
        """Rebuild the full resolution image from the downsampled layers.

        NOTE: This function may modify the first input image (if stride == 1)!
        """
        num_layers = len(self.layers)
        if len(arrays) != num_layers:
            raise TypeError(f"expected {num_layers} arrays, got {len(arrays)}")
        if out is not None and out.shape[-self.ndim:] != self.grid:
            raise TypeError(f"expected shape ending in {self.grid}, got {array.shape}")
        self._compute_alphas()
        # If stride == 1, this will return by reference!
        out = self.layers[0].upscale(arrays[0], out)
        prev = out
        for i in range(1, num_layers):
            curr = self.layers[i].upscale(arrays[i])
            delta = curr - prev[self._relative_slices[i - 1]]
            delta *= self._alphas[i - 1]
            out[self.layers[i].slices] += delta
            prev = curr
        return out
