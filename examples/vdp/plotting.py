#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from examples.vdp.setup import Config

from adaled import AdaLEDStage as Stage, TensorCollection
from adaled.utils.arrays import join_sequences
from adaled.plotting.base import Task, Plotter
from adaled.plotting.external_mpl import confidence_ellipse
from adaled.plotting.plot_record import MergedRecordsPlotter
from adaled.plotting.plots import Axes, Plot, mpl, plt
from adaled.plotting.utils import divider_colorbar, fade_color
from adaled.postprocessing.record import \
        get_cycle_slices, load_and_concat_records, slice_record_trajectory_batch
import adaled

from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import numpy as np

from typing import Dict, Sequence, Tuple, Union
import argparse
import glob

# https://matplotlib.org/stable/api/markers_api.html
LIMIT_CYCLE_MARKERS = 'vvvvvvvvvvvvv'
LIMIT_CYCLE_MARKERS2 = 'ovPxDs<Xd>xp^*'
MICRO_MARKER = 'D'
MACRO_MARKER = 'o'

# Otherwise plotting many trajectories in one go may crash.
mpl.rcParams['agg.path.chunksize'] = 10000

# XLIM = (-3.0, 3.0)
# YLIM = (-3.0, 3.0)
XLIM = (-2.5, 2.5)
YLIM = (-2.5, 2.5)
# ASPECT_RATIO = 1.0
ASPECT_RATIO = 0.7   # Squash y-axis to save vertical space.

FIGWIDTH = float(os.environ.get('VDP_FIGWIDTH', 7.0))


def _plot_limit_cycles(
        ax: Axes,
        limit_cycles: Dict[float, Dict[str, np.ndarray]],
        scatter_kwargs={},
        markers=LIMIT_CYCLE_MARKERS,
        report: bool = False,
        scatter: bool = True,
        labels: bool = True,
        **kwargs):
    """Plot limit cycles.

    Arguments:
        limit_cycles: {mu: {'micro': ..., 'macro': ...}}
    """
    assert len(limit_cycles) < len(markers), \
           f"need {len(limit_cycles)} markers, there are only {len(markers)}"

    for mu, cycles in limit_cycles.items():
        micro = cycles['micro']
        ax.plot(micro[:, 0], micro[:, 1], label=f"$\mu = {mu}$", **kwargs)

    scatter_kwargs = scatter_kwargs.copy()
    for key, value in kwargs.items():
        scatter_kwargs.setdefault(key, value)

    # Note: bbox is inconsistent between png and pdf. To avoid overlap between
    # transparent white bboxes, PDFs need zorder=20 and PNGs zorder=20-i.
    # The bbox stuff below is an attempt to render 1 box for all texts.

    # renderer = ax.get_figure().canvas.get_renderer()
    # bbox = None
    for i, (mu, cycles) in enumerate(limit_cycles.items()):
        macro = cycles['macro']
        if scatter:
            ax.scatter(macro[:, 0], macro[:, 1], zorder=5, marker=markers[i],
                       label=f"macro $\mu = {mu}$", **scatter_kwargs)
        kw = {}
        if report:
            kw['bbox'] = dict(facecolor=(1.0, 1.0, 1.0, 0.8), edgecolor='none',
                              boxstyle='square,pad=0.3', mutation_aspect=0.01)
        if labels:
            t = ax.text(0.0, macro[0, 1] - (0.0 if report else 0.2),
                    f"$\mu = {mu}$", zorder=20,
                    horizontalalignment=('right' if report else 'center'),
                    verticalalignment=('center' if report else 'top'), **kw)
            # bbox_ = t.get_window_extent(renderer=renderer)
            # bbox = bbox_ if bbox is None else bbox_.union([bbox_, bbox])
        print(f"Limit cycle for mu={mu} has ~{len(macro)} macro time steps.")

    # bbox = bbox.transformed(ax.transData.inverted())
    # x, y, w, h = bbox.bounds
    # text_bg = mpl.patches.Rectangle((x, y), w, h, facecolor='yellow', edgecolor='none', zorder=15)
    # ax.add_patch(text_bg)

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_system_phase_space(config: Config, mus: Sequence[float]):
    from examples.vdp.postprocess import integrate_limit_cycle
    def _integ(mu: float, dt: float):
        return integrate_limit_cycle(mu, dt, config.dt_micro, config.circular_motion_system)

    limit_cycles = {
        mu: {
            'micro': _integ(mu, config.dt_micro),
            'macro': _integ(mu, config.dt_rnn)[:-3],
        }
        for mu in mus
    }

    mus_str = '-'.join(f'{mu:.2f}' for mu in mus)
    with Plot(f'report-phase-space-{mus_str}.png:pdf', figsize=(6, 3.5)) as (fig, ax):
        ax: Axes
        ax.set_aspect(ASPECT_RATIO)
        _plot_limit_cycles(ax, limit_cycles, markers=LIMIT_CYCLE_MARKERS2,
                           scatter_kwargs={'s': 15.0}, linewidth=1.0, report=True)


def plot_validation_xy(
        path_or_ax: Union[str, Axes],
        info: adaled.TensorCollection,
        limit_cycles: Dict[float, Dict[str, np.ndarray]],
        title_which: str,
        alpha=None,
        colorbar_ax=None,
        ylabel=None,
        highlighted_cycles: Sequence[int] = [],
        report: bool = False,
        labels: bool = True):
    # https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#directly-creating-a-segmented-colormap-from-a-list
    # colors = ['lawngreen', 'gold', 'darkorange', 'red']
    # colors = ['dodgerblue', 'gold']
    # cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)

    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap = plt.get_cmap('Reds')
    cmap = truncate_colormap(cmap, 0.1, 1.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.2)

    kwargs = {
        'cmap': cmap,
        'norm': norm,
        # 'rasterized': True,
    }

    with Plot(path_or_ax) as (fig, ax):
        ax: Axes

        x_micro = info['x_micro']
        x_macro = info['x_macro']
        assert x_micro.shape == x_macro.shape
        assert x_micro.ndim == 2
        dist = ((x_micro - x_macro) ** 2).sum(axis=-1) ** 0.5
        lines = np.stack([x_micro, x_macro], axis=1)
        lc = mpl.collections.LineCollection(
                lines, array=dist, linewidths=0.7, **kwargs, alpha=alpha)
        ax.add_collection(lc)
        ax.autoscale()

        order = np.argsort(dist)
        # random = np.random.RandomState(seed=12345)
        # random.shuffle(order[:-20])
        x_micro = x_micro[order]
        x_macro = x_macro[order]
        dist = dist[order]

        # Slice into groups of 100 to ensure better z-ordering.
        colors = cmap(norm(dist))
        if alpha is not None:
            colors[:, 3] *= alpha
        n = 100
        sections = list(range(0, len(x_micro), n))
        for i in sections:
            # Use edgecolors='none' to have sharp instead of smooth edges.
            ax.scatter(x_micro[i : i + n, 0], x_micro[i : i + n, 1], s=14.0,
                       marker=MICRO_MARKER, label=("micro" if i == sections[-1] else None),
                       facecolors=colors[i : i + n], edgecolors='none', **kwargs)
            ax.scatter(x_macro[i : i + n, 0], x_macro[i : i + n, 1], s=18.0,
                       marker=MACRO_MARKER, label=("macro" if i == sections[-1] else None),
                       facecolors='none', edgecolors=colors[i : i + n],
                       linewidth=0.7, **kwargs)

        # Highlight selected cycles with an ellipse.
        for idx in highlighted_cycles:
            mid = (x_macro[idx] + x_micro[idx]) * 0.5
            dx = x_macro[idx] - x_micro[idx]
            l = (dx ** 2).sum() ** 0.5
            angle = np.rad2deg(np.arctan2(dx[1], dx[0]))
            ax.add_patch(mpl.patches.Ellipse(
                    mid, 2.2 * l, 0.8 * l, angle,
                    edgecolor='#9437FF', facecolor='none', zorder=1000))

        if colorbar_ax is not False:
            cbar = fig.colorbar(lc, cax=colorbar_ax, label=ylabel)
            cbar.ax.locator_params(nbins=5)

        _plot_limit_cycles(ax, limit_cycles, color='black', linewidth=0.5,
                           scatter=False, labels=labels)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        if not report:
            ax.legend()
            ax.set_title(f"VdP limit cycles and final {title_which} steps")


class _TrajectoriesPlotHelper:
    def __init__(self, fields: TensorCollection, base_size: float = 10.0):
        raw_macro = fields['simulations', 'z', 'raw_macro']
        if raw_macro.ndim == 4:  # Probabilistic?
            raw_macro = raw_macro[..., 0]
        assert raw_macro.ndim == 3 and raw_macro.shape[-1] == 2

        # First dot of each stage is larger.
        stages = fields['metadata', 'stage']
        size = np.full(len(stages), base_size)
        size[0] *= 4
        size[1:][stages[:-1] != stages[1:]] *= 4

        self.raw_macro = raw_macro
        self.x_micro = fields['simulations', 'x', 'micro']
        self.x_macro = fields['simulations', 'x', 'macro']
        self.stages = stages
        self.size = size

    def plot(self,
             ax: Axes,
             ensemble: bool = False,
             ellipses: bool = False,
             skip_first: bool = False,
             **kwargs):
        """
        Arguments:
            skip_first: whether to skip rendering first symbol
        """
        # # Connect micro and macro states.
        # lines = np.stack([x_micro, x_macro], axis=1)
        # lc = mpl.collections.LineCollection(lines, colors='orange', linewidths=0.7)
        # ax.add_collection(lc)
        # ax.autoscale()

        begin = 1 if skip_first else 0
        x_micro = self.x_micro
        x_macro = self.x_macro
        raw_macro = self.raw_macro
        size = self.size[begin:]

        # Use edgecolors='none' to have sharp instead of smooth edges.
        if ensemble:
            for e in range(raw_macro.shape[1]):
                ax.plot(*raw_macro[:, e, :].T, linewidth=0.5, color='gray', zorder=0, **kwargs)
                ax.scatter(*raw_macro[begin:, e, :].T, 0.5 * size,
                           marker=MACRO_MARKER, color='gray', zorder=0, edgecolors='none',
                           label=("ensemble" if e == 0 else None), **kwargs)

        if ellipses:
            for step in raw_macro:
                confidence_ellipse(*step.T, ax, n_std=1.0,
                                   facecolor=fade_color('red', 0.6), zorder=0)

        ax.plot(*x_micro.T, linewidth=0.7, color='blue', zorder=0, **kwargs)
        ax.plot(*x_macro.T, linewidth=0.7, color='red', zorder=0, **kwargs)
        ax.scatter(*x_micro[begin:].T, size, marker=MICRO_MARKER,
                   color='blue', label="micro", zorder=1, edgecolors='none', **kwargs)
        ax.scatter(*x_macro[begin:].T, size, marker=MACRO_MARKER,
                   color='red', label="macro", zorder=1, edgecolors='none', **kwargs)

        ax.set_aspect(ASPECT_RATIO)
        ax.set_facecolor('none')
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def zoomed_plot(
            self,
            ax: Axes, ax_inset: Axes,
            xmin: float,
            xmax: float,
            ymin: float,
            ymax: float,
            padding_factor_x: float = 0.1,
            padding_factor_y: float = 0.1,
            **kwargs):
        xrange = xmax - xmin
        yrange = ymax - ymin

        self.plot(ax_inset, **kwargs)
        ax_inset.set_xlim(xmin - padding_factor_x * xrange, xmax + padding_factor_x * xrange)
        ax_inset.set_ylim(ymin - padding_factor_y * yrange, ymax + padding_factor_y * yrange)
        ax_inset.tick_params(labelleft=False, labelbottom=False)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")


def plot_report_cycle_ensemble(
        path: Union[Axes, str],
        slice_: Tuple[int, int],
        fields: adaled.TensorCollection,
        bounds: Tuple[float, float, float, float],
        zoom: bool = True):
    """Plot one cycle, together with ensemble. Record represents the desired cycle."""

    plot = Plot(path, figsize=(6, 3.5))
    ax = plot.ax

    _stages = fields['metadata', 'stage']
    # is_accepted = Stage.MACRO in _stages
    is_accepted = False  # Don't render macro-only stages for now.
    if is_accepted:
        macro_fields = fields[(_stages == Stage.MACRO).nonzero()[0][0] - 1:]
    fields = fields[(_stages == Stage.WARMUP) | (_stages == Stage.COMPARISON)]

    helper = _TrajectoriesPlotHelper(fields, base_size=8.0)
    helper.plot(ax, ensemble=True)

    if is_accepted:
        helper2 = _TrajectoriesPlotHelper(macro_fields, base_size=6.0)
        helper2.plot(ax, ensemble=False, alpha=0.1, skip_first=True)

    if zoom:
        x_micro = helper.x_micro
        mask = np.arange(len(x_micro)) > (len(x_micro) - 7)
        xmin = x_micro[:, 0][mask].min()
        ymin = x_micro[:, 1][mask].min()
        xmax = x_micro[:, 0][mask].max()
        ymax = x_micro[:, 1][mask].max()
        ax_inset = zoomed_inset_axes(ax, zoom=5, loc='lower center')
        helper.zoomed_plot(ax, ax_inset, xmin, xmax, ymin, ymax, ensemble=True,
                           padding_factor_x=0.3, padding_factor_y=0.3)
        if is_accepted:
            helper2.zoomed_plot(ax, ax_inset, xmin, xmax, ymin, ymax,
                                padding_factor_x=0.3, padding_factor_y=0.3,
                                ensemble=False, alpha=0.1, skip_first=True)
        mark_inset(ax, ax_inset, 2, 3)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.text(0.98, 0.02, f"time step\n#{slice_[0]}+",
            ha='right', va='bottom', transform=ax.transAxes)

    return plot


def plot_report_cycle_ensemble_matrix(
        path: Union[Axes, str],
        slices: np.ndarray,  # of 2 ints
        fields: np.ndarray,  # of TensorCollections
        zoom: np.ndarray,  # of bools
        bounds: Tuple[float, float, float, float]):
    """Plot one cycle per cell."""
    assert slices.ndim == 3, slices.shape
    assert slices.shape[:2] == zoom.shape == fields.shape, \
           (slices.shape, zoom.shape, fields.shape)
    nrows, ncols = slices.shape[:2]
    plot = Plot(path, figsize=(8.0, 8.0), nrows=nrows, ncols=ncols)
    plot.fig.subplots_adjust(wspace=0.05, hspace=0.00)

    for row in range(nrows):
        for col in range(ncols):
            ax: Axes = plot.ax[row, col]
            plot_report_cycle_ensemble(
                    ax, slices[row, col], fields[row, col], bounds, zoom=zoom[row, col])
            if row < nrows - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col >= 1:
                ax.set_ylabel("")
                ax.set_yticklabels([])

    return plot


def plot_bad_accepted_trajectories(
        path: str,
        config: Config,
        bad: adaled.TensorCollection,
        report: bool = True):
    """Plot one bad trajectory."""

    helper = _TrajectoriesPlotHelper(bad)

    x_micro = helper.x_micro
    x_macro = helper.x_macro
    raw_macro = helper.raw_macro

    def finalize(ax: Axes):
        if not report:
            ax.legend()
            ax.set_title(f"VdP bad accepted cycles")
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)

    kwargs = {}
    if report:
        kwargs['figsize'] = (6.0, 4.0)
        path = path.replace('.png', '.png:pdf')
        F = bad['simulations', 'F']
        print(f"    {path}  first time step={bad['metadata', 'timestep'].min()}"
              f"  mu=[{F.min()}..{F.max()}]")

    with Plot(path, **kwargs) as (fig, ax):
        helper.plot(ax, ellipses=True)
        # helper.plot(ax, ensemble=True)
        finalize(ax)

    path = path.replace('.png', '-inset.png').replace('.png:pdf', '.pdf')
    with Plot(path, **kwargs) as (fig, ax):
        ax: Axes
        helper.plot(ax, ellipses=True)

        mask = helper.stages == Stage.MACRO
        error = (((x_micro - x_macro) ** 2).sum(axis=-1)) ** 0.5
        bad_steps = (mask[:-1] & (np.diff(error) > 0.02)).nonzero()[0]
        if len(bad_steps) > 0:
            first_bad_step = [0]
            mask &= np.arange(len(mask)) >= first_bad_step
        mask = np.arange(len(mask)) >= len(mask) - 6
        std = raw_macro.std(axis=1)
        # xmin = min(x_micro[:, 0][mask].min(), (x_macro - std)[:, 0][mask].min())
        # ymin = min(x_micro[:, 1][mask].min(), (x_macro - std)[:, 1][mask].min())
        # xmax = max(x_micro[:, 0][mask].max(), (x_macro + std)[:, 0][mask].max())
        # ymax = max(x_micro[:, 1][mask].max(), (x_macro + std)[:, 1][mask].max())
        xmin = (x_macro - std)[:, 0][mask].min()
        ymin = (x_macro - std)[:, 1][mask].min()
        xmax = (x_macro + std)[:, 0][mask].max()
        ymax = (x_macro + std)[:, 1][mask].max()

        ax_inset = zoomed_inset_axes(ax, zoom=2.0, loc='center left',
                                     # bbox_to_anchor=(0.22, 0.00, 0.4, 0.4),
                                     bbox_to_anchor=(0.22, 0.30, 0.4, 0.4),
                                     bbox_transform=ax.transAxes)
        helper.zoomed_plot(ax, ax_inset, xmin, xmax, ymin, ymax,
                           ellipses=True, padding_factor_y=0.3)
        mark_inset(ax, ax_inset, 2, 3)
        finalize(ax)


def plot_bad_accepted_trajectories_tx(
        path: str, config: Config, bad: Sequence[adaled.TensorCollection]):
    sim = join_sequences([cycle['simulations'] for cycle in bad], gap=np.nan)
    x_micro = sim['x', 'micro']
    x_macro = sim['x', 'macro']
    raw_macro = sim['z', 'raw_macro']
    if raw_macro.ndim == 4:  # Probabilistic?
        raw_macro = raw_macro[..., 0]
    F = sim['F']
    dist = ((x_micro - x_macro) ** 2).sum(axis=-1) ** 0.5

    common = {'markersize': 3.0}
    with Plot(path, nrows=3, ncols=1) as (fig, ax):
        ax: List[Axes]
        ax_unc: Axes = ax[2].twinx()

        for i in range(2):
            for e in range(config.ensemble_size):
                ax[i].plot(raw_macro[:, e, i], 'o-', color='gray', linewidth=0.5,
                           label=("ensemble" if e == 0 else None), markersize=2.0)
            ax[i].plot(x_micro[:, i], 'o-', color='green', label="micro", **common)
            ax[i].plot(x_macro[:, i], 'o-', color='red', label="macro", **common)

        ax[2].plot(F, 'o-', color='purple', label="$\mu$", **common)
        F_jump = (np.abs(F[:-1] - F[1:]) > 1e-6).nonzero()[0]
        for t in F_jump:
            ax[2].axvline(t, color='purple', linewidth=0.5)

        ax_unc.plot((2 * sim['uncertainty']) ** 0.5, 'o-', color='black', label="sqrt(2*uncertainty)", **common)
        ax_unc.plot(dist, 'o-', color='red', label="error $E$", **common)
        ax_unc.set_yscale('log')
        # ax_unc.plot(raw_macro.std(axis=1).sum(axis=-1), label="std_x + std_y")

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()

        lines1, labels1 = ax[2].get_legend_handles_labels()
        lines2, labels2 = ax_unc.get_legend_handles_labels()
        ax[0].legend()
        ax[2].legend(lines1 + lines2, labels1 + labels2)


def plot_report_validation(
        path: str,
        accepted: adaled.TensorCollection,
        limit_cycles: Dict[float, Dict[str, np.ndarray]],
        highlighted_cycles: Sequence[int] = []):

    fig = plt.figure(figsize=(FIGWIDTH, 3.0), dpi=200)
    with Plot(path, fig=fig, ax=fig.get_axes()) as (fig, axes):
        # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots#38940369
        grid: Sequence[Axes] = ImageGrid(
                fig, 111,  # as in plt.subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=0.0,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05,
                aspect=ASPECT_RATIO)

        ax1, ax2 = grid
        ax1.set_aspect(ASPECT_RATIO)
        ax2.set_aspect(ASPECT_RATIO)
        plot_validation_xy(
                ax1, accepted['last_cmp'], limit_cycles, "", report=True,
                colorbar_ax=False)
        plot_validation_xy(
                ax2, accepted['last_macro'], limit_cycles, "", report=True,
                colorbar_ax=ax2.cax, ylabel="macro step error $E$",
                highlighted_cycles=highlighted_cycles)


def plot_report_validation_scatter(
        path: str,
        accepted_info: adaled.TensorCollection):
    valid_error = accepted_info['final_validation_error']
    length = accepted_info['length']
    with Plot(path, figsize=(6.0, 4.0)) as (fig, ax):
        ax: Axes
        # ax.scatter(length, valid_error, s=size, c=accepted_info['num_macro_steps'])
        dist = (valid_error * 2) ** 0.5
        x = accepted_info['num_macro_steps']
        y = dist / length
        # size = dist
        size = 4 + 30 * (accepted_info['max_F'] - accepted_info['min_F'])
        # size = 100 * dist
        # color = 4 + 10 * (accepted_info['max_F'] - accepted_info['min_F'])
        # color = accepted_info['last_cmp', 'uncertainty']
        color = accepted_info['max_F']
        # alpha = 0.2 + 0.8 * accepted_info['start_timestep'] / accepted_info['start_timestep'].max()
        alpha = None
        im = ax.scatter(x, y, s=size, c=color, edgecolor='none', alpha=alpha, zorder=10)
        divider_colorbar(fig, im, ax)
        ax.set_xlim(0, x.max())
        ax.set_ylim(bottom=3e-5, top=3e-1)
        ax.set_xlabel("trajectory spatial length")
        ax.set_ylabel(r"final macro step validation error $E_{valid}$")
        ax.set_yscale('log')
        # ax.grid(axis='y', which='minor', linewidth=0.5, color='lightgray')
        # ax.grid(axis='y', which='major')


class StandalonePlotter(Plotter):
    # Kind of standalone, still needs dt_micro and dt_macro.

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--custom-mu', type=float, nargs='*',
                            help="if specified, plot limit cycles for these mus")

    def tasks_system_custom(self):
        config: Config = self.context.load_config()
        if self.args.custom_mu:
            yield Task(plot_system_phase_space, config, self.args.custom_mu)


class VDPPlotter(Plotter):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument
        add('--cycles', type=int, nargs='*', default=[],
            help="cycles to plot with ensemble")
        add('--cycles-filter-start-coord', type=float, nargs=4,
            metavar=('X0', 'X1', 'Y0', 'Y1'),
            default=(0.6, 1.4, 0.6, 1.4),
            help="filter cycles according to the start coordinate")
        add('--cycles-limits', type=float, nargs=4,
            metavar=('X0', 'X1', 'Y0', 'Y1'),
            default=(-2.5, 2.5, -1.7, 1.4),
            help="bounds for the cycle ensemble plots")
        add('--cycles-matrix', type=int, nargs=2,
            metavar=('NROWS', 'NCOLS'), default=(4, 2),
            help="optionally render cycles in a matrix")
        add('--cycles-zoom-since', type=int, default=4,
            help="at which cycle to start adding zoom insets")
        add('--cycles-accepted', type=str,
            choices=('accepted', 'rejected', 'any'), default='any',
            help="accepted cycles, rejected or any?")
        add('--highlighted-cycles', type=int, nargs='*', default=[],
            help="which cycles (sorted by error) to highlight with an ellipse")

    def set_up(self):
        if hasattr(self, 'config'):
            if self.record is None:
                self.skip()
            return
        self.config: Config = self.context.load_config()
        self.diagnostics: adaled.AdaLEDDiagnostics = adaled.load('diagnostics-000.pt')

        paths = sorted(glob.glob('record-0*.pt'))
        record = load_and_concat_records(paths)
        if record['fields'].keys():
            self.record = slice_record_trajectory_batch(record, s=0)
        else:
            self.record = None
            self.skip()

    def _load_postprocessed(self):
        if hasattr(self, 'data'):
            return self.data
        try:
            self.data = adaled.load('postprocessed-macro-cycles.pt')
        except Exception as e:
            raise e.__class__("Did you run ./postprocess.py?")
        if self.data.get('version') != 3:
            raise RuntimeError("rerun ./postprocess.py")
        return self.data

    def tasks_validation_xy(self, report=False):
        prefix = 'report-validation-xy' if report else 'validation-xy'
        ext = 'png:pdf' if report else 'png'
        data = self._load_postprocessed()
        accepted = data['accepted_cycles_info']
        rejected = data['rejected_last_cmp']
        limit_cycles = data['limit_cycles']
        if len(rejected) > 0:
            rejected = TensorCollection(
                    x_micro=rejected['simulations', 'x', 'micro', :, 0, :],
                    x_macro=rejected['simulations', 'x', 'macro', :, 0, :])
            yield Task(plot_validation_xy, f'{prefix}-rejected-last-cmp.{ext}',
                       rejected, limit_cycles, "rejected comparison",
                       alpha=np.linspace(0.1, 1.0, len(rejected)), report=report)
        if len(accepted) > 0:
            yield Task(plot_validation_xy, f'{prefix}-accepted-last-cmp.{ext}',
                       accepted['last_cmp'], limit_cycles, "accepted comparison", report=report)
            yield Task(plot_validation_xy, f'{prefix}-accepted-last-macro.{ext}',
                       accepted['last_macro'], limit_cycles, "accepted macro", report=report)
        else:
            print("Skipping accepted trajectories plot, no accepted trajectories.")

    def tasks_report_cycle_ensemble(self):
        cycle_slices = get_cycle_slices(self.record['fields', 'metadata', 'stage'])

        matrix: Optional[Tuple[int, int]] = self.args.cycles_matrix
        cycle_ids: List[int] = self.args.cycles
        if not cycle_ids and not matrix:
            return

        if not cycle_ids and matrix:
            if self.args.cycles_accepted == 'accepted':
                accepted = [i for i, (b, e) in enumerate(cycle_slices)
                            if Stage.MACRO in self.record['fields', 'metadata', 'stage', b:e]]
                cycle_ids = accepted[:matrix[0] * matrix[1]]
            elif self.args.cycles_accepted == 'rejected':
                rejected = [i for i, (b, e) in enumerate(cycle_slices)
                            if Stage.MACRO not in self.record['fields', 'metadata', 'stage', b:e]]
                cycle_ids = rejected[:matrix[0] * matrix[1]]
            else:
                cycle_ids = np.arange(matrix[0] * matrix[1])

        if self.args.cycles_filter_start_coord:
            x0, x1, y0, y1 = self.args.cycles_filter_start_coord
            xy = self.record['fields', 'simulations', 'x', 'micro', cycle_slices[:, 0]]
            is_ok = (x0 <= xy[:, 0]) & (xy[:, 0] <= x1) \
                  & (y0 <= xy[:, 1]) & (xy[:, 1] <= y1)
        else:
            is_ok = np.full(len(cycle_slices), True)

        final_ids = []
        final_slices = []
        final_fields = []
        for cycle_id in cycle_ids:
            cid = max(cycle_id, final_ids[-1] + 1 if final_ids else 0)
            while cid < len(is_ok) and not is_ok[cid]:
                cid += 1

            if cid < len(is_ok):
                s = cycle_slices[cid]
                final_ids.append(cid)
                final_slices.append(s)
                final_fields.append(self.record['fields', s[0]:s[1]])
                xy = final_fields[-1]['simulations', 'x', 'micro', 0]
                print(f"Plotting cycle {cid} (instead of {cycle_id}) starting at {xy}.")
            else:
                final_ids.append(final_ids[-1])
                final_slices.append(final_slices[-1])
                final_fields.append(final_fields[-1])
                # FIXME: do not plot anything then...
                print(f"Nothing left to plot in the plot matrix, repeating the previous cell.")

        if matrix:
            nrows, ncols = matrix
            ids = '-'.join(map(str, final_ids))

            fields = np.array(final_fields, dtype=object).reshape(ncols, nrows).T
            zoom = np.arange(len(final_slices)) >= self.args.cycles_zoom_since
            zoom = zoom.reshape(ncols, nrows).T
            yield Task(plot_report_cycle_ensemble_matrix, f'report-cycle-{ids}-matrix.png:pdf',
                       np.moveaxis(np.array(final_slices).reshape(ncols, nrows, 2), 0, 1),
                       fields, zoom, self.args.cycles_limits)
        else:
            for i, s, fields in zip(final_ids, final_slices, final_fields):
                yield Task(plot_report_cycle_ensemble, f'report-cycle-{i:06d}.png',
                           s, cycle_fields, self.args.cycles_limits)

    def tasks_bad_accepted_trajectories(self):
        bad = self._load_postprocessed()['bad_accepted_cycles']
        if len(bad) == 0:
            print("No bad accepted cycles.")
            return
        if len(bad) > 10:
            print("Plotting only first 10 bad accepted cycles.")
            bad = bad[:10]
        for i, cycle in enumerate(bad):
            yield Task(plot_bad_accepted_trajectories,
                       f'accepted-bad-xy-{i:02d}.png', self.config, bad[i])

    def tasks_bad_accepted_trajectories_tx(self):
        bad = self._load_postprocessed()['bad_accepted_cycles']
        if len(bad) == 0:
            print("No bad accepted cycles.")
            return
        if len(bad) > 5:
            print("Plotting only first 10 bad accepted cycles.")
            bad = bad[:5]
        yield Task(plot_bad_accepted_trajectories_tx,
                   'accepted-bad-tx.png', self.config, bad)

    def tasks_last_movie_frame(self):
        from examples.vdp.movie import load_record_and_plot_movie
        yield Task(load_record_and_plot_movie, 'movie-last.png',
                   self.config, frame_begin=-1)

    def tasks_report_validation_xy(self):
        data = self._load_postprocessed()
        yield Task(plot_report_validation, 'report-validation-final.png:pdf',
                   data['accepted_cycles_info'], data['limit_cycles'],
                   self.args.highlighted_cycles)

    def tasks_report_validation_scatter(self):
        data = self._load_postprocessed()
        if len(data['accepted_cycles_info']) == 0:
            return
        yield Task(plot_report_validation_scatter, 'report-validation-scatter.png:pdf',
                   data['accepted_cycles_info'])


class MergedRecordsPlotterEx(MergedRecordsPlotter):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument
        add('--mu-title', type=str, default=r'$\mu$')

    def plot_report_utilization_and_F(self, *args, **kwargs):
        plot = super().plot_report_utilization_and_F(
                *args, F_title=self.args.mu_title, **kwargs)
        plot.ax1.set_ylim(0.9, 3.1)
        plot.ax2.set_ylim(0.0, 0.8)
        return plot


if __name__ == '__main__':
    from adaled.plotting.all import main
    main([StandalonePlotter, MergedRecordsPlotterEx, VDPPlotter])
