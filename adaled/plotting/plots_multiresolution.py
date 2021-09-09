from adaled.transformers.multiresolution import Multiresolution, FinalizedLayer
from adaled.plotting.plots import Axes
from adaled.plotting.utils import fade_color, get_default_rgba
import adaled

from matplotlib.patches import Rectangle
import numpy as np

from typing import Optional, Sequence, Tuple

# DEFAULT_COLORS = [(0.9, 0.9, 0.9), (1.0, 0.0, 0.0), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)]
# DEFAULT_COLORS = [(0.95, 0.95, 0.95), (1.0, 0.7, 0.7), (1.0, 0.0, 0.0), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)]
DEFAULT_COLORS = [fade_color(c, 0.85)[:3] for c in get_default_rgba(5)]

def plot_multiresolution_2d(
        ax: Axes,
        mr: Multiresolution,
        colors: Optional[Sequence[Tuple[float, float, float]]] = None,
        dims: Tuple[int, int] = (0, 1),
        report: bool = True,
        padding: float = 0.005):
    """Plot multiresolution setup. Plot alpha (weights) as an image and mark
    layer boundaries."""
    if colors is None:
        colors = DEFAULT_COLORS

    ndim = mr.layers[0].ndim

    arrays = []
    for c, l in zip(colors, mr.layers):
        c = np.array(c, dtype=np.float32)
        c = c[(slice(None),) + (None,) * ndim]
        c = np.broadcast_to(c, (3,) + l.downscaled_shape)
        # Torch complains about the non-writeable flag created by broadcast_to,
        # set it back to writeable (this cannot really be done later).
        c.setflags(write=True)
        arrays.append(c)
    image = mr.rebuild_unsafe(arrays)
    image = np.moveaxis(image, 0, -1)

    assert len(dims) == 2 and dims[0] < dims[1], dims
    if ndim > 2:
        # Average along the axes we don't show.
        image = image.mean(tuple(ndim - d - 1 for d in range(ndim) if d not in dims))

    ax: Axes
    ax.imshow(image)

    def _2d(x: Tuple[int, ...]):
        return tuple(x[d] for d in dims)

    padding = padding * max(_2d(mr.grid))
    for i, l in enumerate(mr.layers):
        margin = l.alpha_margin_cells
        begin = _2d(l.begin)
        size = _2d(l.size)
        end = _2d(l.end)
        stride = _2d(l.stride)
        if i > 0:
            ax.add_patch(Rectangle(begin, *size, edgecolor='black', facecolor='none'))
            ax.add_patch(Rectangle(
                    (begin[0] + margin, begin[1] + margin),
                    size[0] - 2 * margin,
                    size[1] - 2 * margin,
                    edgecolor='black', linestyle='--', facecolor='none'))
        if report:
            # if i > 0:
            #     ax.text(begin[0], begin[1] - padding, str(begin), ha='center', va='bottom')
            #     ax.text(end[0], end[1] + padding, str(end), ha='center', va='top')
            cx = (begin[0] + end[0]) / 2
            cy = (begin[1] + end[1]) / 2
            if i > 0:
                ax.annotate("", (prev_begin[0], cy), (begin[0], cy), arrowprops={'arrowstyle':'<->'})
                ax.annotate(str(begin[0] - prev_begin[0]),
                            ((begin[0] + prev_begin[0]) / 2, cy - padding), ha='center')
                ax.annotate("", (cx, prev_begin[1]), (cx, begin[1]), arrowprops={'arrowstyle':'<->'})
                ax.annotate(str(begin[1] - prev_begin[1]),
                            (cx + padding, (begin[1] + prev_begin[1]) / 2), ha='left', va='center')

            text0 = str(size[0]) if stride[0] == 1 else f"{size[0]} (1/{stride[0]}x)"
            text1 = str(size[1]) if stride[1] == 1 else f"{size[1]}\n(1/{stride[1]}x)"
            if i == 0:
                ax.text(cx, begin[1] - padding, text0, ha='center', va='bottom')
            else:
                ax.text(cx, end[1] + padding, text0, ha='center', va='top')
            ax.text(end[0] + padding, cy, text1, ha='left', ma='center', va='center')
            step = max(_2d(mr.grid)) // 8
            ax.set_xticks(range(0, mr.grid[dims[0]] + step, step))
            ax.set_yticks(range(0, mr.grid[dims[1]] + step, step))
        else:
            ax.text(*begin, f"{l.begin}-{l.end} size={l.size} stride={l.stride}",
                    horizontalalignment='left', va='top')

        prev_begin = begin
        prev_end = end


def plot_multiresolution_1d_contributions(
        ax: Axes,
        path: str,
        mr: Multiresolution,
        xlabel: str,
        colors):
    """Plot multiresolution training and reconstruction bounds as a 1D slice."""
    assert mr.ndim == 1, \
           "expected 1d multiresolution, use dim_mask argument " \
           "if you have a 2D or a 3D system"
    layers = mr.layers
    L = mr.grid[0]

    reconstruct_weights = mr.compute_reconstruction_contributions()

    bottom = np.zeros(L)
    x = 0.5 + np.arange(L)
    x[0] = 0.0
    x[-1] = L
    for i, (color, layer) in enumerate(zip(colors, layers)):
        layer: FinalizedLayer
        top = bottom + reconstruct_weights[i]
        ax.fill_between(x, bottom, top, alpha=0.2)
        bottom = top
        if i > 0:
            ax.axvspan(layer.begin[0], layer.end[0], color=color, alpha=0.1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("contribution")
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1)
    ax.grid()
