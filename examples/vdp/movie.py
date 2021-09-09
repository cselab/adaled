#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from examples.vdp.setup import Config
from examples.vdp.plotting import XLIM, YLIM

from adaled import AdaLEDStage, TensorCollection
from adaled.plotting.base import Task, Plotter
from adaled.plotting.plot_diagnostics import \
        DiagnosticsAnimationPlot, HistogramAnimationPlot
from adaled.plotting.plots import Axes, Figure, mpl, plt
from adaled.postprocessing.record import \
        load_and_concat_records, slice_record_trajectory_batch
import adaled.plotting.utils as utils
import adaled

import numpy as np

from typing import Optional
import argparse
import glob


# TODO: Move channel-related plots to plotting/plots_0d.py.
def plot_movie(
        path: str,
        config: Optional[Config],
        record: TensorCollection,
        diagnostics: TensorCollection,
        frame_begin: int = 0,
        frame_end: Optional[int] = None,
        frame_skip: int = 1,
        fps: int = 30,
        tpf: int = 100,  # Timesteps per frame.
        trajectory_window: int = 400,
        macro_roll_window: int = 2000,
        workers: Optional[int] = None,
        *,
        report: bool = False):
    """
    Make a movie of VdP micro and macro trajectories.

    If report is True, the plot will exclude the dataset histogram plots and
    show fewer lines in the bottom plot.
    """
    fields = record['fields']
    # Per-step uncertainty is very noisy and not very informative, skip it.
    del fields['simulations']['uncertainty']

    num_frames = len(fields) // tpf
    t = np.arange(len(fields))
    colors = utils.get_default_rgba(n=4)
    colors = [colors[0], colors[3]]  # Blue and red by default.
    sim = fields['simulations']

    # FIXME: x_micro seems to be one step ahead at the beginning of the
    #        simulation. Later the time shift disappears!
    # x_micro = sim['x', 'micro']
    # x_macro = sim['x', 'macro']
    # x_ensemble = sim['z', 'raw_macro'][:, :, :, 0]  # Only the per-RNN mean.
    x_micro = sim['x', 'micro'][:-1]
    x_macro = sim['x', 'macro'][1:]
    x_ensemble = sim['z', 'raw_macro'][1:, :, :, 0]  # Only the per-RNN mean.
    fields = fields[1:]
    sim = sim[1:]

    # Ignore reconstructed parts of the trajectory.
    mask_stage = 1 << fields['metadata', 'stage']
    x_micro[(mask_stage & adaled.AdaLEDStage.MASK_MICRO) == 0] = np.nan
    x_macro[(mask_stage & adaled.AdaLEDStage.MASK_MACRO) == 0] = np.nan
    x_ensemble[(mask_stage & adaled.AdaLEDStage.MASK_MACRO) == 0] = np.nan

    if report:
        fig: Figure = plt.figure(constrained_layout=True, figsize=(9.0, 7.0), dpi=150)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1])
    else:
        fig: Figure = plt.figure(constrained_layout=True, figsize=(12.0, 8.0))
        gs = fig.add_gridspec(nrows=2, ncols=2,
                              height_ratios=[2, 1], width_ratios=[3, 1])
        gs_histo = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1])
        ax_histo = [fig.add_subplot(gs_histo[i]) for i in range(2)]
    ax0: Axes = fig.add_subplot(gs[0, 0])
    ax1: Axes = fig.add_subplot(gs[1, :])

    # Top left: the phase space.
    kwargs0 = {'fade': 1.0, 'color': colors[0], 'linewidth': 2.0}
    kwargs1 = {'fade': 1.0, 'color': colors[1], 'linewidth': 2.0}
    kwargs2 = {'fade': 1.0, 'color': 'gray', 'linewidth': 1.0}

    def add_line(points, kwargs, marker: str):
        line, colors = utils.add_faded_line(ax0, *points.T, return_colors=True, **kwargs)
        # scatter = ax0.scatter(*points.T, marker=marker, c=colors)
        # return line, scatter
        return line

    def update_line(ls, points, kwargs):
        # line, scatter = ls
        kwargs = dict(kwargs)
        fade = kwargs.pop('fade') * len(points) / trajectory_window
        line = ls
        line, colors = utils.add_faded_line(
                ax0, *points.T, **kwargs, lc=line, return_colors=True, fade=fade)
        # scatter.set_offsets(points)
        # scatter.set_color(colors)
        return [line]

    ensemble_ls = [
        add_line(x_ensemble[:trajectory_window, i, :], kwargs2, marker='o')
        for i in range(config.ensemble_size)
    ]
    micro_ls = add_line(x_micro[:trajectory_window], kwargs0, marker='D')
    macro_ls = add_line(x_macro[:trajectory_window], kwargs1, marker='o')
    ax0.set_xlim(*XLIM)
    ax0.set_ylim(*YLIM)

    # Top right: dataset histogram.
    hc = config.dataset_histograms
    if report and hc:
        hh = diagnostics['dataset', 'train', 'histograms']
        histograms = HistogramAnimationPlot(
                ax_histo,
                # Macro histograms include data over all ensemble members.
                [[hh['F']], [hh['cmp_error'], hh['latest_loss_macro_mse']]],
                [[hc['F']], [hc['cmp_error'], hc['latest_loss_macro_mse']]],
                [[None], ["cmp MSE", "macro MSE"]],
                ["$\mu$", "error"],
                ylabel="training sample count")

    # Bottom: fraction macro steps, training loss, MSE, VDP mu (F).
    _last_cycle = np.searchsorted(diagnostics['start_timestep'],
                                  (num_frames + 1) * tpf)
    diagnostics = diagnostics[:_last_cycle + 1]
    if report:
        del diagnostics['losses']['macro_train']  # Do not show training error.
    diagnostics_plot = DiagnosticsAnimationPlot(ax1, fields, diagnostics,
            error_label="MSE", F_label="$\mu$", legend=(not report),
            criteria=config.criteria, macro_roll_window=macro_roll_window)
    ax1.set_ylabel("error and uncertainty")
    cycle_t = diagnostics['start_timestep']

    def update(frame):
        end = (frame + 1) * tpf
        begin = max(end - trajectory_window, 0)

        last_cycle = getattr(update, 'last_cycle', -1)
        update.last_cycle = cycle = np.searchsorted(cycle_t[:-1], end)

        updated = []
        for i in range(config.ensemble_size):
            updated += update_line(ensemble_ls[i], x_ensemble[begin:end, i], kwargs2)
        updated += update_line(micro_ls, x_micro[begin:end], kwargs0)
        updated += update_line(macro_ls, x_macro[begin:end], kwargs1)
        updated += diagnostics_plot.update(end, cycle, last_cycle)
        if report and hc:
            updated += histograms.update(end, cycle,last_cycle)

        ax0.set_title(f"AdaLED cycle #{cycle}")
        return updated

    from adaled.plotting.animation import parallelized_animation
    frame_range = range(num_frames)[frame_begin:frame_end:frame_skip]
    parallelized_animation(path, fig, update, frame_range,
                           interval=1000 // fps, blit=True, workers=workers)
    plt.close(fig)
    print(f"Saved {path}")


def load_record_and_plot_movie(
        path: str, config: Config, max_frames: int = -1, **kwargs):
    """Load the first trajectory in the record files and plot a movie (or a
    single image, depending on the path extension)."""
    paths = sorted(glob.glob('record-0*.pt'))
    record = load_and_concat_records(paths, max_frames=max_frames)
    record = slice_record_trajectory_batch(record, s=0)
    diagnostics = adaled.load('diagnostics-000.pt')
    plot_movie(path, config, record, diagnostics, **kwargs)


class VDPMoviePlotter(Plotter):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument
        add('--num-frames', type=int, default=-1)
        add('--report', default=False, action='store_true', help="make the movie cleaner")
        add('--tpf', type=int, default=100, help="timesteps per frame")
        add('--slice', type=int, nargs=3, default=(0, None, 1), help="slice BEGIN END STEP")
        add('--trajectory-window', type=int, default=400,
            help="trajectory trace length, in time steps")
        add('--output', type=str, default='movie.mp4')

    def tasks_movie(self):
        args = self.args
        config: Config = adaled.load('config.pt')
        yield Task(load_record_and_plot_movie, args.output, config,
                   max_frames=args.num_frames, workers=args.jobs,
                   report=args.report, tpf=args.tpf,
                   frame_begin=args.slice[0], frame_end=args.slice[1],
                   frame_skip=args.slice[2],
                   trajectory_window=args.trajectory_window)


if __name__ == '__main__':
    VDPMoviePlotter.main()
