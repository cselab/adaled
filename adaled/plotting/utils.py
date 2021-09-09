from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Sequence
import itertools

Axes = mpl.axes.Axes

# TODO: rename to faded_line
def add_faded_line(
        ax: Axes,
        x: Sequence[float],
        y: Sequence[float],
        fade: float = 0.5,
        color=None,
        lc: Optional[LineCollection] = None,
        *,
        return_colors: bool = False,
        **kwargs):
    """Plot a faded line with given x and y values.

    If return_colors is True, returns (lc, colors), where `lc` is the
    LineCollection object. Otherwise, returns lc. The colors have same length
    as the input.
    """
    if color is None:
        # This is not ideal, but there is probably no easy robust way of
        # getting the next color without affecting the cycler.
        props = next(ax._get_lines.prop_cycler)
        color = props['color']
        props.update(kwargs)  # kwargs have priority.
        kwargs = props
        del props['color']

    color = np.asarray(mpl.colors.to_rgba(color))

    points = np.stack([x, y], axis=-1)
    points = points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = gradual_fade(color, len(segments) + 1, fade)
    # Note that if the line starts or ends with nans that the colors
    # may look overly faded.
    if lc is None:
        lc = LineCollection(segments, colors=colors[:-1], **kwargs)
        ax.add_collection(lc)
    else:
        lc.set_segments(segments)
        lc.set_colors(colors[:-1])
    return lc, colors if return_colors else lc


def concatenate_and_interleave_trajectories(
        trajectories: Sequence[np.ndarray], separator=np.nan):
    """Concatenate trajectories and add NaNs inbetween."""
    separator = np.asarray(separator)
    parts = []
    for i, t in enumerate(trajectories):
        t = np.asarray(t)
        if i > 0:
            parts.append(np.broadcast_to(separator, (1,) + t.shape[1:]))
        parts.append(t)

    return np.concatenate(parts)


def divider_colorbar(fig, im, ax: Axes, size: str = '5%', pad: float = 0.05, **kwargs):
    """
    Divide the axes and put the colorbar on the right.

    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

    Arguments:
        kwargs: forwarded to `fig.colorbar`

    Returns the tuple (divider axes, cax, colobar).
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, **kwargs)
    return (divider, cax, cbar)


def fake_log_yscale(ax: Axes, minor: bool = True):
    """Emulate log scale ticks and labels.

    Useful for violin plots that don't support log scale."""
    # https://stackoverflow.com/questions/60131839/violin-plot-troubles-in-python-on-log-scale#60132262
    ymin, ymax = ax.get_ylim()
    large_yrange = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax.yaxis.set_ticks([y for y in large_yrange if ymin <= y <= ymax])
    if minor:
        shifts = np.log10(np.arange(1, 11))
        minor_ticks = np.concatenate([p + shifts for p in large_yrange])
        ax.yaxis.set_ticks([y for y in minor_ticks if ymin <= y <= ymax], minor=True)


def fake_log_yscale_violinplot(
        ax: Axes,
        data: Sequence[np.ndarray],
        x: Sequence[float],
        showmeans: bool = True,
        *,
        s: float = 100,
        marker: str = '_',
        color=None,
        **kwargs):
    """Fixes for violinplot which does not support log scale."""
    if not color:
        color = next(ax._get_lines.prop_cycler)['color']

    violin: dict = ax.violinplot([np.log10(col) for col in data], x, showmeans=False, **kwargs)
    for key, parts in violin.items():
        try:
            iter(parts)
        except TypeError:
            parts = (parts,)
        for part in parts:
            if hasattr(part, 'set_facecolor'):
                part.set_facecolor(color)
            if key != 'bodies' and hasattr(part, 'set_edgecolor'):
                part.set_edgecolor(color)

    for pc in violin['bodies']:
        pc.set_facecolor(color)

    if showmeans:
        ax.scatter(x, [np.log10(col.mean()) for col in data], color=color, s=s, marker=marker)

    return violin


def get_default_rgba(n: int) -> np.ndarray:
    """Return first n RGBA colors from the default matplotlib color cycle.

    Returns a numpy array of shape (n, 4).
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = np.asarray(mpl.colors.to_rgba_array(colors))
    k = len(colors)
    return np.tile(colors, ((n + k - 1) // k, 1))[:n]


def darken_color(color, darken: float = 0.5):
    if isinstance(color, np.ndarray) and color.ndim > 1:
        color = mpl.colors.to_rgba_array(color)
    else:
        color = np.array(mpl.colors.to_rgba(color))  # Copy.
    color[..., :3] *= 1 - darken
    return color


def fade_color(color, fade: float = 0.5):
    """Fade color towards white."""
    color = np.asarray(mpl.colors.to_rgba(color))
    white = _white_like(color)
    return fade * white + (1 - fade) * color


def _white_like(color: np.ndarray):
    """If color is RGB, return white RGB. If it is RGBA, return white with the
    same alpha channel."""
    if color.shape[-1] == 3:
        return np.array([1, 1, 1])
    elif color.shape[-1] == 4:
        white = np.ones_like(color)
        white[..., 3] = color[..., 3]
        return white
    else:
        raise ValueError(f"unexpected shape {color.shape}")


def gradual_fade(
        color: Sequence[float],
        size: int,
        fade: float = 0.5):
    """Starting from a faded `color` by `fade` fraction, compute a linear
    gradient towards the original color `color`.

    Arguments:
        color: rgb or rgba color
        size: number of colors to compute
        fade: white color intensity

    Returns:
        A `(size, d)` array, where `d` is 3 or 4, matching the `color` argument.
    """
    assert 3 <= len(color) <= 4
    return np.linspace(fade_color(color, fade), color, size)


def human_formatter(digits: int):
    symbols = ['', 'K', 'M', 'G', 'T', 'P']

    def human_format(num, pos):
        # https://stackoverflow.com/questions/61330427/set-y-axis-in-millions
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return f'{num:.{digits}f}{symbols[magnitude]}'

    return human_format


def periodic_line_x(
        ax: Axes,
        x: Sequence[float],
        y: Sequence[float],
        label: Optional[str] = None,
        linestyle: Optional[str] = None,
        periodic_linestyle: Optional[str] = None,
        periodic: bool = True,
        x_extent: Optional[float] = None,
        **kwargs):
    """Plot an x-y line, repeat it shifted to the left and to the right."""
    ax.plot(x, y, label=label, linestyle=linestyle, **kwargs)
    if periodic:
        if x_extent is None:
            x_extent = (x[-1] - x[0]) * len(x) / (len(x) - 1)
        ax.plot(np.append(x - x_extent, x[0]),
                np.append(y, y[0]),
                linestyle=periodic_linestyle, **kwargs)
        ax.plot(np.insert(x + x_extent, 0, x[-1]),
                np.insert(y, 0, y[-1]),
                linestyle=periodic_linestyle, **kwargs)
