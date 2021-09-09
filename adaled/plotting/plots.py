import adaled.plotting.utils as utils

from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import os
import sys


# Disallow importing as `import plots`, allow only
# `import adaled.plotting.plots`, to avoid importing twice.
assert 'adaled' in sys.modules
import adaled
assert not hasattr(adaled, '_plotting_plots_loaded'), \
        "use `import adaled.plotting.plots`, not `import plots`"
adaled._plotting_plots_loaded = True

Axes = mpl.axes.Axes        # Alias.
Figure = mpl.figure.Figure  # Alias.
AxesArrayLike = Union[Sequence[Sequence[Axes]], Sequence[Axes], Axes]

def rescale_to(x, *, min, max):
    """Rescale given array to values between [min, max]."""
    xmin = x.min()
    xmax = x.max()
    scale = (max - min) / (xmax - xmin)
    shift = min - scale * xmin
    out = x * scale + shift
    assert abs(out.min() - min) < 1e-3, (out.min(), min)
    assert abs(out.max() - max) < 1e-3, (out.max(), max)
    return out


class Plot:
    """Context manager for plotting, automatically saves the plot on exit.

    Example:
        >>> with Plot(path='filename.png') as (fig, ax):
        >>>     ax.plot(...)
        >>>     ...
        >>> # Automatically saves the plot to `filename.png` after `with`.
    """
    def __init__(self,
                 path_or_axes: Optional[Union[str, AxesArrayLike]] = None,
                 nrows: int = 1,
                 ncols: int = 1,
                 *,
                 fig: Optional[Figure] = None,
                 ax: Optional[AxesArrayLike] = None,
                 # figsize: Tuple[float, float] = (6.4, 4.8),  # <-- default
                 figsize: Tuple[float, float] = (16.0, 9.0),
                 suptitle: str = None,
                 title: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 xlim: tuple = None,
                 ylim: tuple = None,
                 legend: bool = False,
                 grid: bool = False,
                 **kwargs):
        """Create a plot context manager that automatically saves the plot to
        the given path, if provided.

        Any unrecognized kwargs are forwarded to `plt.subplots()`.
        """
        if isinstance(path_or_axes, str):
            self.path: str = path_or_axes
            if fig is None and ax is None:
                fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
            elif fig is not None and ax is None:
                ax = fig.get_axes()
            elif fig is None and ax is not None:
                # Or is there a way to get fig from ax?
                raise TypeError("fig must be provided if ax is provided")
        else:
            self.path = None
            if not isinstance(path_or_axes, Axes):
                ax = np.asarray(path_or_axes, dtype=object)
                assert all(isinstance(ax_, Axes) for ax_ in ax.ravel()), ax
                fig = ax.ravel()[0].figure
            else:
                ax = path_or_axes
                fig = ax.figure
        fig: Figure
        ax: AxesArrayLike
        self.fig = fig
        self.ax = ax

        if suptitle:
            fig.suptitle(suptitle)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # These things have to be enabled after plotting everything.
        self.xlim = xlim
        self.ylim = ylim
        self.legend = legend
        self.grid = grid
        self._finalized = False

    def __enter__(self) -> Tuple[Figure, Axes]:
        return self.fig, self.ax

    def __exit__(self, type, value, traceback):
        if type is None:
            self.finalize()

    def finalize(self):
        """Set the axes limits and legend if specified at construct time.
        Save the file if path was specified."""
        assert not self._finalized, "finalize() already called"
        self._finalized = True
        fig, ax = self.fig, self.ax
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        if self.legend:
            ax.legend()

        if self.grid:
            if isinstance(ax, Axes):
                ax = np.array([ax])
            for ax_ in ax.ravel():
                ax_.grid()

        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            base, exts = os.path.splitext(self.path)
            # Hacky way to support multiple extensions... A better solution
            # would be to use bash curly braces syntax.
            for ext in exts[1:].split(':'):
                path = f'{base}.{ext}'
                if ext == 'tex':
                    import tikzplotlib
                    tikzplotlib.save(path, figure=fig)
                else:
                    fig.savefig(path, bbox_inches='tight')
                print(f"Plot saved to {path}.")
            plt.close(fig)

    _finalize_task = finalize  # See Task.__call__.


class MultidimensionalStateMatrixPlot(Plot):
    def __init__(self, path, title, subplot_func,
                 datasets, dataset_labels, axes_labels, **kwargs):
        dim = datasets[0].shape[-1]
        super().__init__(path, nrows=dim - 1, ncols=dim - 1, squeeze=False, **kwargs)
        fig, axs = self.fig, self.ax

        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        for row in range(dim - 1):
            for col in range(dim - 1):
                ax: mpl.axes.Axes = axs[row, col]
                ix = col
                iy = dim - 1 - row
                if ix >= iy:
                    fig.delaxes(ax)
                    continue
                try:
                    selected_columns = datasets[..., ix:iy+1:iy-ix]
                except TypeError:
                    selected_columns = [dataset[..., ix:iy+1:iy-ix]
                                        for dataset in datasets]
                assert selected_columns[0].shape[-1] == 2
                subplot_func(ax, selected_columns, dataset_labels)
                if col == 0:
                    ax.set_ylabel(axes_labels[iy])
                else:
                    ax.set_yticklabels([])
                if ix == iy - 1:
                    ax.set_xlabel(axes_labels[ix])
                else:
                    ax.set_xticklabels([])
                if row == 0 and col == 0:
                    ax.legend()
                ax.set_axisbelow(True)  # Put grid behind the lines.
                ax.grid()



class MatrixScatterPlot(Plot):
    """Plots one or more sets of D-dimensional points in a (D-1)x(D-1) matrix
    of scatter plots, optionally connecting different sets with lines.

    Each scatter plot plots two out of D point coordinates.

    This plot can be used for example to visualize the reconstruction loss,
    where one set represents the original dataset and one set the reconstructed
    dataset.
    """
    def __init__(self,
                 path: str,
                 title: str,
                 point_sets: 'arraylike',
                 labels: Sequence[str],
                 styles: Iterable[Dict[str, Any]],
                 axes_labels: Sequence[str],
                 *,
                 subtitle: Optional[str] = None,
                 connect_point_sets: bool = True,
                 connect_style: Dict[str, Any] = {'linewidth': 0.7, 'alpha': 0.8},
                 **kwargs):
        """
        Arguments:
            path: target path
            title: plot (sup)title
            point_sets: a 3-dimensional array (point set, point, coord)
            labels: labels of point sets
            styles: custom styles applied to each point set scatter plot
            axes_labels: labels of axes, one per coord dimension
            subtitle: optional subtitle to apply, if `None`, it will be automatically added
            connect_point_sets: whether to connect points of different point sets
            connect_style: keyword arguments to pass to LineCollection
            **kwargs: forwarded to Plot
        """
        point_sets = np.asarray([np.asarray(point_set) for point_set in point_sets])
        assert point_sets.ndim == 3, point_sets.shape
        dim = point_sets.shape[-1]
        super().__init__(path, dim - 1, dim - 1, **kwargs)
        fig, axs = self.fig, self.ax

        if dim == 1:
            colors = rescale_to(point_sets[0, :, 0], min=0.0, max=1.0)
            colors = plt.get_cmap()(colors)
        elif dim == 2:
            colors = np.stack([
                rescale_to(point_sets[0, :, 0], min=0.1, max=0.8),
                rescale_to(point_sets[0, :, 1], min=0.1, max=0.8),
                0.5,
            ], axis=-1)
        elif dim >= 3:
            colors = np.stack([
                rescale_to(point_sets[0, :, 0], min=0.1, max=0.8),
                rescale_to(point_sets[0, :, 1], min=0.1, max=0.8),
                rescale_to(point_sets[0, :, 2], min=0.1, max=0.8),
            ], axis=-1)

        if dim >= 4:
            sizes = rescale_to(point_sets[0, :, 1], min=5.0, max=40.0)
        else:
            sizes = 30.0

        # Precompute styles, as likely we won't be able to iterate through
        # `styles` multiple times.
        styles = [style for _, style in zip(point_sets, styles)]

        if subtitle is None:
            subtitle = f"\nRGB = {labels[0]} $x_{{0..2}}$, size = $x_3$ (if available)"
        fig.suptitle(title + subtitle)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        for row in range(dim - 1):
            for col in range(dim - 1):
                ax: mpl.axes.Axes = axs[row, col]
                ix = col
                iy = dim - 1 - row
                if ix >= iy:
                    fig.delaxes(ax)
                    continue
                if connect_point_sets:
                    # lines is an array (line index, vertex index, coord),
                    # constructed from the array (point set index, point index, coords).
                    # First we convert D-dim state to a 2D point (without
                    # copying!), then we swap first two axes.
                    lines = point_sets[:, :, ix:iy+1:iy-ix]
                    lines = np.moveaxis(lines, 0, 1)
                    assert lines.shape[-1] == 2  # (x, y) coords
                    lc = LineCollection(lines, colors=colors, **connect_style)
                    ax.add_collection(lc)
                for points, label, style in zip(point_sets, labels, styles):
                    style = style.copy()
                    # This is a tricky/hacky way to specify whether to fill
                    # markers or not.
                    if style.get('facecolors', -1) is None:
                        style['facecolors'] = colors
                    ax.scatter(points[..., ix], points[..., iy], sizes, label=label,
                               edgecolors=colors, **style)
                if col == 0:
                    ax.set_ylabel(axes_labels[iy])
                else:
                    ax.set_yticklabels([])
                if ix == iy - 1:
                    ax.set_xlabel(axes_labels[ix])
                else:
                    ax.set_xticklabels([])
                if row == 0 and col == 0:
                    ax.legend()
                ax.set_axisbelow(True)  # Put grid behind the lines.
                ax.grid()


class MatrixTrajectoryPlot(Plot):
    """Plot D-dim state trajectories as a (D-1)x(D-1) matrix of plots between
    two state variables.
    """
    def __init__(self,
                 path: str,
                 title: str,
                 trajectories: Sequence[np.ndarray],
                 trajectory_labels: Sequence[str],
                 axes_labels: Sequence[str],
                 *,
                 fade: float = 0.5,
                 **kwargs):
        """
        Arguments:
            path: target file
            title: title of the whole plot
            trajectories: list of (time step, state var) tensors, lines to plot
            trajectory_labels: names of corresponding trajectories
            axes_labels: names of axes
            kwargs: forwarded to Plot
        """
        assert len(trajectories) == len(trajectory_labels)
        assert len(axes_labels) == trajectories[0].shape[-1]

        dim = trajectories[0].shape[-1]
        super().__init__(path, dim - 1, dim - 1, squeeze=False, **kwargs)
        fig, axs = self.fig, self.ax

        fig: mpl.figure.Figure
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        for row in range(dim - 1):
            for col in range(dim - 1):
                ax: mpl.axes.Axes = axs[row, col]
                ix = col
                iy = dim - 1 - row
                if ix >= iy:
                    fig.delaxes(ax)
                    continue
                for traj, label in zip(trajectories, trajectory_labels):
                    utils.add_faded_line(ax, traj[..., ix], traj[..., iy],
                                         fade=fade, label=label)
                ax.autoscale()  # Required with `add_faded_line`.
                if col == 0:
                    ax.set_ylabel(axes_labels[iy])
                else:
                    ax.set_yticklabels([])
                if ix == iy - 1:
                    ax.set_xlabel(axes_labels[ix])
                else:
                    ax.set_xticklabels([])
                if row == 0 and col == 0:
                    ax.legend()
                ax.grid()


class LinePlotMatrix(Plot):
    """Plots u_{k,row,col}(x) as columns of matrix of line plots.

    The different `k` lines are plotted on top of each other.
    """
    def __init__(self,
                 path: str,
                 title: str,
                 datasets: np.ndarray,
                 xlabel: str,
                 channel_names: Sequence[str],
                 dataset_names: Sequence[str],
                 *,
                 x: Optional[np.ndarray] = None,
                 shared_ylim: bool = False,
                 **kwargs):
        assert datasets.ndim == 4, datasets.shape
        num_datasets, num_samples, num_channels, width = datasets.shape
        assert len(channel_names) == num_channels
        super().__init__(path, nrows=num_samples, ncols=num_channels, squeeze=False,
                         suptitle=title, **kwargs)
        fig, axs_matrix = self.fig, self.ax

        if x is None:
            x = np.arange(width)

        # Based on the following tutorial:
        # https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
        # Change from (dataset, sample, channel, x) to (sample, channel, dataset, x).
        reordered_datasets = np.moveaxis(datasets, 0, 2)
        ymin = [+np.inf] * num_channels
        ymax = [-np.inf] * num_channels
        for row, (row_data, axs_row) in enumerate(zip(reordered_datasets, axs_matrix)):
            for col, (ax, lines_data, name) in \
                    enumerate(zip(axs_row, row_data, channel_names)):
                ax: Axes
                for line_data in lines_data:
                    ax.plot(x, line_data)
                    if shared_ylim:
                        ymin[col] = min(ymin[col], line_data.min())
                        ymax[col] = max(ymax[col], line_data.max())
                if row == 0:
                    ax.set_title(name)
                if row == num_samples - 1:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xticklabels([])

        if shared_ylim:
            for axs_row in axs_matrix:
                for y0, y1, ax in zip(ymin, ymax, axs_row):
                    ax.set_ylim(y0, y1)


class ImagePlotMatrix(Plot):
    """Plot a matrix of images."""
    def __init__(self,
                 path: str,
                 title: str,
                 images: Sequence[Sequence[np.ndarray]],
                 image_titles: Sequence[Sequence[str]],
                 *,
                 xlabel: str = "x",
                 ylabel: str = "t",
                 kwargs: Dict[str, Sequence[Sequence[Any]]] = {},
                 **kwargs_):
        super().__init__(path, nrows=len(images), ncols=len(images[0]),
                         squeeze=False, suptitle=title, **kwargs_)

        kwargs = {
            k: np.broadcast_to(np.asarray(v, dtype=object), self.ax.shape)
            for k, v in kwargs.items()
        }
        fig: Figure = self.fig
        for row, (ax_row, image_row, name_row) in \
                enumerate(zip(self.ax, images, image_titles)):
            for col, (ax, image, name) in \
                    enumerate(zip(ax_row, image_row, name_row)):
                ax: Axes
                cell_kwargs = {k: v[row, col] for k, v in kwargs.items()}
                im = ax.imshow(image, origin='lower', aspect='auto',
                               interpolation='bilinear', **cell_kwargs)
                fig.colorbar(im, ax=ax)
                ax.set_title(name)
                if row == len(images) - 1:
                    ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(ylabel)
                # If there are too many channels, clear the clutter by not 
                # showing y ticks on columns other than the first one.
                if len(ax_row) > 2 and col > 0:
                    ax.set_yticklabels([])


class ImageErrorPlotMatrix(ImagePlotMatrix):
    """Plot two columns and their difference.

    Automatically computes the third column (non-negative absolute error) and
    the common vmin and vmax for the first two columns.
    """
    def __init__(self,
                 path: str,
                 title: str,
                 images: Sequence[Sequence[np.ndarray]],
                 image_titles: Sequence[Sequence[str]],
                 *,
                 kwargs: Optional[Dict] = {},
                 **kwargs_):
        num_channels = len(images)
        assert len(images[0]) == 2, "expected left and right"
        new_images = []
        kwargs = kwargs.copy()
        vmins = []
        vmaxs = []
        for lhs, rhs in images:
            new_images.append((lhs, rhs, np.abs(rhs - lhs)))
            vmin = min(lhs.min(), rhs.min())
            vmax = min(lhs.max(), rhs.max())
            vmins.append((vmin, vmin, None))
            # vmaxs.append((vmax, vmax, max(-vmin, vmax)))
            vmaxs.append((vmax, vmax, None))
        kwargs.setdefault('cmap', [[None, None, 'Reds']])
        kwargs.setdefault('vmin', vmins)
        kwargs.setdefault('vmax', vmaxs)
        super().__init__(path, title, new_images, image_titles,
                         kwargs=kwargs, **kwargs_)
