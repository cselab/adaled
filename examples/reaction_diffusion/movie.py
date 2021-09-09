#!/usr/bin/env python3

from typing import Optional, Sequence
import argparse
import glob

import numpy as np

from adaled import TensorCollection
from adaled.plotting.base import Task, Plotter
from adaled.plotting.plot_diagnostics import DiagnosticsAnimationPlot
from adaled.plotting.plots import Axes, Figure, mpl, plt
from adaled.plotting.plots_2d import ChannelComparison2DAnimationPlot
from adaled.postprocessing.record import LazyRecordLoader
import adaled

from .setup import Config


def plot_movie(path: str, config: Config, report: bool = False, **kwargs):
    paths = sorted(glob.glob(f'record-0*.h5'))
    if not paths:
        print("No records found, skipping the movie (or movie frame) plot.")
        return

    if not report:
        try:
            diagnostics = adaled.load(f'diagnostics-000.pt')
        except FileNotFoundError as e:
            print("WARNING:", str(e))
            diagnostics = None
    else:
        diagnostics = None

    _plot_movie(path, config, paths, diagnostics, report=report, **kwargs)


def _plot_movie(
        path: str,
        config: Config,
        input_paths: Sequence[str],
        diagnostics: Optional[TensorCollection],
        layer: Optional[int] = None,
        num_frames: int = -1,
        frame_begin: int = 0,
        frame_end: Optional[int] = None,
        frame_skip: Optional[int] = None,
        frames: Optional[Sequence[int]] = None,
        fps: int = 30,
        workers: Optional[int] = None,
        *,
        bitrate: int = -1,
        clean: bool = False,
        dpi: int = 100,
        no_micro_if_macro: bool = False,
        report: bool = False):
    loader = LazyRecordLoader(input_paths, num_frames)
    fields = loader.small_fields
    metadata = fields['metadata']

    num_channels = 2

    fig: Figure = plt.figure(constrained_layout=True, figsize=(12, 9.0), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 2])
    gs_channels = mpl.gridspec.GridSpecFromSubplotSpec(num_channels, 3, subplot_spec=gs[0])
    ax_channels = [[fig.add_subplot(gs_channels[i, j]) for j in range(3)]
                   for i in range(num_channels)]

    norm1 = mpl.colors.Normalize(vmin=-1.0, vmax=+1.0)
    # norm2 = mpl.colors.Normalize(vmin=0.0, vmax=0.1)
    norm2 = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    norm_abs = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    kwargs_matrix = {
        # [micro, macro, error, abs error]
        'norm': [
            [norm1, norm1, norm2, norm_abs],
            [norm1, norm1, norm2, norm_abs],
        ],
        'cmap': [['RdBu', 'RdBu', 'Reds', 'Reds']] * 3,
    }

    state_plot = ChannelComparison2DAnimationPlot(
            ax_channels, loader, kwargs_matrix=kwargs_matrix,
            channel_names=["$u$", "$v$"],
            no_micro_if_macro=no_micro_if_macro, nan_to_num=0.0)

    if diagnostics:
        ax1: Axes = fig.add_subplot(gs[1])
        _last_cycle = np.searchsorted(diagnostics['start_timestep'],
                                      metadata['timestep'][-1])
        diagnostics = diagnostics[:_last_cycle + 1]
        diagnostics_plot = DiagnosticsAnimationPlot(
                ax1, fields, diagnostics,
                error_label="MSE", F_label=r"$d_1$/$d_2$/$\beta$",
                macro_roll_window=500, criteria=config.criteria,
                legend=True)
        diagnostics_plot.ax_macro.set_ylim(0.0, 1.0)
        cycle_t = diagnostics['start_timestep']

    def update(frame):
        updated = []

        timestep = metadata['timestep'][frame]

        if diagnostics:
            last_cycle = getattr(update, 'last_cycle', -1)
            update.last_cycle = cycle = np.searchsorted(cycle_t[:-1], timestep)
            updated += diagnostics_plot.update(frame, cycle, last_cycle)
        else:
            cycle = "?"

        updated += state_plot.update(frame)

        if not report:
            fig.suptitle(f"AdaLED cycle #{cycle}, timestep #{timestep}")

        return updated

    from adaled.plotting.animation import parallelized_animation
    if frame_skip is None:
        frame_skip = config.recorder.x_every
        if frame_skip > 1:
            print(f"Plotting only frames for which the real state is "
                  f"available (x_every={frame_skip}).")
    if frames is None:
        frames = range(len(metadata))[frame_begin:frame_end:frame_skip]
    parallelized_animation(path, fig, update, frames,
                           interval=1000 // fps, blit=True, workers=workers,
                           bitrate=bitrate)
    plt.close(fig)
    print(f"Saved {path}")



class MoviePlotter(Plotter):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        # TODO: These settings should be a part of the library.
        add = parser.add_argument
        add('--num-frames', type=int, default=-1)
        add('--slice', type=int, nargs=2, default=(0, None))
        add('--skip', type=int)
        add('--fps', type=int, default=30, help="frame rate")
        add('--dpi', type=int, default=100, help="DPI")
        add('--no-micro-if-macro', default=False, action='store_true',
            help="disable micro view on macro-only steps, "
                 "useful for validation runs with always_run_micro=1")
        add('--output-prefix', type=str, default='')
        add('--bitrate', type=int, default=-1,
            help="bitrate in kb/s, default -1 (automatic)")

    def tasks_movie(self, layer: Optional[int] = None):
        args = self.args
        config: Config = self.context.load_config()
        yield Task(
                plot_movie,
                f'movie{args.output_prefix}.mp4',
                config,
                num_frames=args.num_frames,
                frame_begin=args.slice[0], frame_end=args.slice[1],
                frame_skip=args.skip, workers=args.jobs,
                _task_no_parallel=True, fps=args.fps,
                no_micro_if_macro=args.no_micro_if_macro,
                bitrate=args.bitrate, dpi=args.dpi)


if __name__ == '__main__':
    MoviePlotter.main()
