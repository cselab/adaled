#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adaled.backends import TensorCollection
from adaled.led.diagnostics import AdaLEDStage
from adaled.plotting.base import Plotter, Task, TaskCase
from adaled.plotting.plots import Axes, Figure, Plot, mpl
from adaled.plotting.utils import darken_color, fade_color, get_default_rgba
from adaled.postprocessing.record import \
        compute_macro_utilization, get_cycle_slices, \
        load_record, normalize_record, \
        slice_record, slice_record_trajectory_batch
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.glob_ import glob_ex
import adaled
import adaled.backends as backends
import adaled.plotting.plots as plots
import adaled.plotting.utils as utils

from matplotlib.collections import LineCollection
import numpy as np
import torch

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import math
import pickle
import re

FIGWIDTH = float(os.environ.get('ADALED_FIGWIDTH', 8.0))

@dataclass
class StageInfo:
    color: str
    label: str


STAGE_INFOS = {
    AdaLEDStage.WARMUP: StageInfo('orange', "warm-up"),
    AdaLEDStage.COMPARISON: StageInfo('yellow', "comparison"),
    AdaLEDStage.MACRO: StageInfo('green', "macro"),
    AdaLEDStage.RELAXATION: StageInfo('black', "relaxation"),
    AdaLEDStage.MICRO: StageInfo('red', "micro"),
}

def _is_array(x):
    """Returns True if `x` is array-like and not a dictionary."""
    try:
        x.ndim
    except:
        return False
    else:
        return True


def add_stage_shaded_regions(ax, x, stages, stage_infos=STAGE_INFOS, alpha=0.15):
    assert len(x) == len(stages), (x.shape, stages.shape)
    i = 0
    n = len(x)
    seen = set()
    while i < n:
        last_stage = stages[i]
        j = i
        while j < n and stages[j] == last_stage:
            j += 1
        info = stage_infos.get(last_stage)
        seen.add(last_stage)
        if info is not None:
            label = None if last_stage in seen else info.label
            ax.axvspan(x[i], x[j if j < n else n - 1],
                       color=info.color, label=label, alpha=alpha, lw=0.0)
        i = j


def find_per_simulation_record_files(
        path_fmts: List[str],
        exclude: List[str] = []) -> Dict[str, List[str]]:
    number_re = re.compile(r'\d+')
    output = defaultdict(list)
    for path in sorted(glob_ex(path_fmts, exclude=exclude)):
        numbers = number_re.findall(os.path.splitext(path)[0])
        sim_id = numbers[0] if len(numbers) >= 2 else ''
        output[sim_id].append(path)
    return output


def plot_t_vs_states(
        path: str,
        title: str,
        stages: np.ndarray,
        timesteps: np.ndarray,
        states_list: List[np.ndarray],
        labels_list: List[List[str]]):
    assert states_list[0].ndim == 2
    linestyles = ('-', '--', ':', '-.')
    with Plot(path) as (fig, ax):
        ax: Axes
        add_stage_shaded_regions(ax, timesteps, stages)
        # kwargs = next(ax._get_lines.prop_cycler)['color']
        for i, (states, labels, linestyle) in \
                enumerate(zip(states_list, labels_list, linestyles)):
            colors = utils.get_default_rgba(len(labels))
            colors = utils.darken_color(colors, darken=i / (1 + len(states_list)))
            for state, label, color in zip(states.T, labels, colors):
                ax.plot(timesteps, state.T, label=label, color=color, linestyle=linestyle)
        ax.set_title(title)
        ax.set_xlabel("time step")
        ax.set_xlim(timesteps.min(), round(timesteps.max(), -1))
        ax.grid()
        ax.legend(loc='lower right')


def plot_probabilistic_rnn_ensemble_t_mu_sigma(
        path: str,
        title: str,
        label_fmt: Optional[str],
        x: np.ndarray,  # Timestep.
        mu: np.ndarray,
        sigma2: np.ndarray,
        uncertainty: np.ndarray,
        *,
        scale_std: float = 1.0,
        z_labels: List[str],
        z_shift: float = 0.0,
        report: bool = False,
        ylim: Optional[Tuple[float, float]] = None):
    """Plot one latent state variable per row, in each row plot multiple lines
    with their uncertainties.

    Arguments:
        scale_std: Scale standard deviation. Currently does not scale the
                   difference between individual RNNs.
    """
    # x = x[::2]
    # mu = mu[::2]
    # sigma2 = sigma2[::2]
    # uncertainty = uncertainty[::2]

    assert mu.ndim == 3 and sigma2.ndim == 3
    linestyles = ('-', '--', ':', '-.')
    num_timesteps, ensemble_size, latent_dim = mu.shape
    sigma = np.sqrt(sigma2)
    x_min = x.min()
    x_max = round(x.max(), -1)

    if report:
        plot = Plot(path, figsize=(FIGWIDTH, 4.5))
        axs: Sequence[Axes] = [plot.ax]
    else:
        plot = Plot(path, nrows=(latent_dim + 1), squeeze=False)
        axs: Sequence[Axes] = plot.ax[:, 0]

    if ensemble_size == 1:
        # This is an agreggate already, better plot everything in purple to denote latent space.
        colors = ['purple']
    else:
        colors = get_default_rgba(ensemble_size)
    for i in range(latent_dim):
        ax: Axes = axs[0] if report else axs[i]
        for j in range(ensemble_size):
            mean = mu[:, j, i]
            if report:
                y_origin = -i * z_shift
                mean = mean + y_origin
                ax.axhline(y_origin, color='gray', zorder=-100, linewidth=0.5)
            std = sigma[:, j, i] * scale_std
            label = label_fmt.format(j=j) if label_fmt else None
            ax.plot(x, mean, label=label, color=colors[j], linewidth=1.0)
            ax.fill_between(x, mean - std, mean + std, alpha=0.3,
                            facecolor=colors[j])
        ax.set_xlim(x_min, x_max)
        if ylim:
            ax.set_ylim(*ylim)
        if report:
            ax.text(x_min, y_origin, z_labels[i] + " ", ha='right', va='center')
        else:
            ax.set_xticklabels([])
            ax.set_ylabel(z_labels[i]);

    if not report:
        for ax in axs:
            ax.grid()
        ax = axs[-1]
        for i in range(latent_dim):
            for j in range(ensemble_size):
                label = z_labels[i] if j == 0 else None
                ax.plot(x, sigma[:, j, i], label=label,
                        color=colors[j], linewidth=1.0,
                        linestyle=linestyles[i % len(linestyles)])
        ax.plot(x, np.sqrt(uncertainty), label="mean", color='k', linewidth=1.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("sqrt(uncertainty)")
        ax.set_yscale('log')
        ax.legend(ncol=(2 if latent_dim >= 5 else 1),
                  labelspacing=0.0, handleheight=0.0)

        if label_fmt:
            axs[0].legend(ncol=(2 if latent_dim >= 6 else 1),
                          labelspacing=0.0, handleheight=0.0)
    else:
        axs[-1].set_yticks([])

    axs[-1].set_xlabel("time step")

    fig = plot.fig
    fig.tight_layout()
    if report:
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
    else:
        fig.subplots_adjust(top=0.95, hspace=0.1)
        fig.suptitle(title)
    return plot


def plot_execution_time(
        path_linear: str,
        path_log: str,
        fields: TensorCollection,
        smooth_sigma: float = 500.0):
    data = fields['metadata']
    if 'timestep' not in data:
        return  # Nothing to plot.

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        # Skip first NaN.
        avg_execution_time = gaussian_filter1d(data['execution_time'][1:], smooth_sigma)

    for path_ in [path_linear, path_log]:
        with Plot(path_) as (fig, ax):
            ax: Axes
            timestep = data['timestep']
            if 'execution_time' in data:
                ax.plot(timestep, data['execution_time'],
                        label="execution time [s]", linewidth=0.7)
                if smooth_sigma > 0:
                    ax.plot(timestep[1:], avg_execution_time,
                            label="execution time (smooth) [s]")
            ax.set_xlim(timestep.min(), timestep.max())
            ax.set_xlabel("time step")
            # Skip the first NaN and the very slow first time step.
            # Skip also other NaNs, appearing after each restart.
            # FIXME: Execution time should never be nan.
            ymax = 1.1 * np.nanmax(data['execution_time'][2:])
            if path_ is path_linear:
                ax.set_ylim(bottom=0.0, top=ymax)
            else:
                ax.set_ylim(bottom=1e-3, top=ymax)
                ax.set_yscale('log')
            ax.grid()
            ax.legend()


@dataclass
class RecordedTrajectoryPlotterConfig(DataclassMixin):
    path: str
    num_trajectories: int = 0
    slice: Optional[Tuple[int, int]] = None


class PerRecordTrajectoryPlotter(Plotter):
    # FIXME: This should be somehow merged with the merged records plotter.

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument_group('trajectory').add_argument
        add('--traj-report', action='store_true', default=False,
            help="cleaner plots, also output as PDF")
        add('--traj-paths', type=str, action='append',
            default=['record-*latest*.pt', 'record-*latest*.h5'],
            help="trajectory path patterns")
        add('--traj-batch-size', type=int, default=1,
            help="limit batch size, 0 to plot whole batch")
        add('--traj-slice', nargs=3, type=int,
            metavar=("BEGIN", "END", "STEP"),
            help="specify trajectory slice")
        add('--traj-z-slice', nargs=3, type=int, default=(0, None, None),
            metavar=("BEGIN", "END", "STEP"),
            help="which latent state elements to plot")
        add('--traj-z-shift', type=float, default=0.3,
            help="shift to use between consecutive lines in EEG-like --traj-report plots")
        add('--traj-scale-std', type=float, default=1.0,
            help="scale standard deviation to make it more visible "
                 "(does not make the ensemble more distant)")
        add('--traj-ylim', nargs=2, type=float, default=None,
            metavar=("BOTTOM", "TOP"),
            help="manually specify ylim for all plots")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        args = self.args
        impls = []
        path_groups = find_per_simulation_record_files(args.traj_paths)
        for key, paths in path_groups.items():
            for path in paths:
                config = RecordedTrajectoryPlotterConfig(
                        path=path,
                        num_trajectories=args.traj_batch_size,
                        slice=args.traj_slice)
                impls.append(RecordedTrajectoryPlotterImpl(config, args, self.context))

        self.impls = impls

    def task_subcases(self):  # <-- do not rename this!
        return self.impls


class MergedRecordsPlotter(Plotter):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('record')
        add = group.add_argument
        add('--record-slice', nargs=3, type=int,
            metavar=('BEGIN', 'END', 'STRIDE'), help="record slice")
        return group

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        begin, end, stride = self.args.record_slice or (0, -1, 1)

        path_groups = find_per_simulation_record_files(
                ['record-0*.pt', 'record-0*.h5'], exclude=['record-*latest*'])
        self.records = {}
        for key, paths in path_groups.items():
            record = load_record(paths, self.record_load_filter)
            fields = record.get('fields')
            if fields is not None and fields.keys() and len(fields) > 0:
                self.records[key] = slice_record(record, slice(begin, end, stride))

    def record_load_filter(self, keys: Tuple[str, ...], d):
        if len(keys) >= 2 and keys[0] == 'fields':
            if keys[1] == 'simulations':
                # For now, accept only 0-dimensional and 1-dimensional F.
                return d if keys[2] == 'F' and len(d.shape) <= 3 else None
            elif keys[1] == 'metadata':
                return d
            else:
                return None
        return d

    def make_path(self, path: str):
        return path

    def tasks_execution_time(self):
        """Plot small data like execution time, stage etc in merged plots."""
        for key, record in self.records.items():
            yield Task(plot_execution_time,
                       self.make_path(f'record-{key}-execution-time.png'),
                       self.make_path(f'record-{key}-execution-time-log.png'),
                       record['fields'])

    def tasks_report_utilization_F(self):
        for key, record in self.records.items():
            yield Task(self.plot_report_utilization_and_F,
                       self.make_path(f'report-{key}-utilization-F.png:pdf'),
                       record['fields'])

    def plot_report_utilization_and_F(
            self,
            path: str,
            fields: TensorCollection,
            stride: int = 50,
            macro_smoothing: float = 1000,
            F_title: str = "F",
            F_color: str = 'blue',
            util_color: str = 'green'):
        from scipy.ndimage import gaussian_filter1d

        t = fields['metadata', 'timestep']
        stages = fields['metadata', 'stage']
        is_macro = stages == int(AdaLEDStage.MACRO)
        dark_util_color = mpl.colors.to_hex(darken_color(util_color, 0.2))

        averaged_util = gaussian_filter1d(is_macro.astype(np.float32), macro_smoothing)
        cycle_t, cycle_utilization = compute_macro_utilization(t, stages)

        plot = Plot(path, figsize=(FIGWIDTH, 1.35))

        ax1 = plot.ax
        F = fields['simulations', 'F'][::stride]
        # Merge batch dimension and potential component dimension of (1D) F.
        # TODO: Improve the plot and the arguments to work with multicomponent F.
        #       Maybe create a common "style" class instead of passing F_title,
        #       F_color etc all the time?
        F = F.reshape(len(F), -1)
        ax1.plot(t[::stride], F, linewidth=1.5, color=F_color)
        ax1.set_xlim(0, round(t[-1], -3))
        ax1.set_xlabel("time step")
        ax1.set_ylabel(F_title, color=F_color)
        ax1.spines['left'].set_edgecolor(F_color)
        ax1.spines['right'].set_edgecolor(dark_util_color)
        ax1.tick_params(axis='y', colors=F_color)

        ax2: Axes = ax1.twinx()
        # Manually set number of bins to 5, because we want to have a very thin plot.
        ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax2.plot(t[::stride], averaged_util[::stride], color=util_color)
        # ax2.stairs(cycle_utilization, cycle_t, color=fade_color(util_color, 0.75), fill=True)
        ax2.stairs(cycle_utilization, cycle_t, color=util_color, alpha=0.25, fill=True)
        ax2.set_ylabel("macro utilization", color=dark_util_color)
        ax2.set_ylim(0.0, 1.0)
        ax2.tick_params(axis='y', colors=dark_util_color)
        ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

        # ax1.grid(axis='x', color='lightgray')
        ax2.grid(axis='y', color='lightgray')
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.set_facecolor('none')
        ax1.set_axisbelow(True)
        ax2.set_frame_on(False)
        ax2.set_axisbelow(True)

        plot.ax1 = ax1
        plot.ax2 = ax2

        return plot



class RecordedTrajectoryPlotterImpl(TaskCase):
    MAX_CHANNELS = 5

    def __init__(self, config: RecordedTrajectoryPlotterConfig, args, context):
        self.config = config
        self.args = args
        self.context = context
        self.simdata = None

    def set_up(self):
        if self.simdata is not None:
            return
        record = normalize_record(adaled.load(self.config.path))
        version, fields, self.output_prefix = _process_config(record, self.config)
        simdata = fields['simulations'].cpu_numpy()

        # Since version 5, the record does not anymore use (trajectory,
        # timestep) ordering but (timestep, trajectory). Swap to simplify
        # iterating over different trajectories.
        self.simdata = simdata = simdata.map(lambda x: np.moveaxis(x, 0, 1))

        z_slice = slice(*self.args.traj_z_slice)
        self.x_micro = simdata['x', 'micro']
        self.z_micro = simdata['z', 'micro', :, :, z_slice]
        self.x_macro = simdata['x', 'macro']
        self.z_macro = simdata['z', 'macro', :, :, z_slice]
        self.raw_macro = simdata['z', 'raw_macro', :, :, :, z_slice]
        self.z_labels = [f"$z_{{{i}}}$" for i in range(simdata['z', 'micro'].shape[-1])[z_slice]]

        self.timestep = fields['metadata', 'timestep']
        self.stage = fields['metadata', 'stage']
        self.F = simdata['F']

        self.context.init(self.x_micro.shape[2:], self.z_macro.shape[2:])

    def make_path(self, suffix, trajectory_index: Optional[int] = None):
        """Generate plot path by combining the prefix setting, the given suffix
        and optionally the trajectory index."""
        if trajectory_index is not None:
            suffix, ext = os.path.splitext(suffix)
            path = f'{self.output_prefix}{suffix}-T{trajectory_index}{ext}'
        else:
            path = self.output_prefix + suffix
        if self.args.traj_report and not path.endswith('pdf'):
            path += ':pdf'
        return path

    def tasks_vector_x(self):
        """Plots when a single x is a vector."""
        if not _is_array(self.x_micro) or self.x_micro.ndim != 3:
            self.skip("only when x is a 1D state")

        print("Real space state is a vector, plotting t-x.")
        xlabels = self.context.xlabels
        for i, (x_micro, x_macro) in enumerate(zip(self.x_micro, self.x_macro)):
            yield Task(plot_t_vs_states, self.make_path('t-x.png', i),
                       "real state", self.stage, self.timestep,
                       [x_micro, x_macro],
                       [xlabels, [l + " (reconstructed)" for l in xlabels]])
            yield Task(plots.MatrixTrajectoryPlot,
                       self.make_path('phase-space-x.png', i),
                       "Real phase space",
                       [x_micro, x_macro],
                       [
                           "micro trajectory (all stages)",
                           "reconstructed macro trajectory (where available)",
                       ],
                       xlabels)
            nans = 0 * (x_micro + x_macro)  # 0 * nan == nan
            yield Task(plots.MatrixTrajectoryPlot,
                       self.make_path('phase-space-x-part.png', i),
                       "Real micro trajectory and reconstructed macro "
                       "during warmup and comparison stages",
                       [x_micro + nans, x_macro + nans],
                       ["micro trajectory", "reconstructed macro trajectory"],
                       xlabels)

    def tasks_vector_z(self):
        """Plots when a single z is a vector."""
        if not _is_array(self.z_macro) or self.z_macro.ndim != 3:
            self.skip("only when z is a 1D state")

        print("Latent space state is a vector, plotting t-z.")
        zlabels = self.context.zlabels
        timestep = self.timestep
        for i, (z_micro, z_macro, uncertainty) in \
                enumerate(zip(self.z_micro, self.z_macro, self.simdata['uncertainty'])):
            if z_micro is not None:
                zs = [z_micro, z_macro]
                labels = [[l + " (transformed)" for l in zlabels], zlabels]
                titles = ["transformed micro trajectory", "latent macro trajectory"]
            else:
                zs = [z_macro]
                labels = [zlabels]
                titles = ["latent macro trajectory"]
            yield Task(plot_t_vs_states, self.make_path('t-z.png', i),
                       "latent state", self.stage, self.timestep, zs, labels)
            # yield Task(plots.MatrixTrajectoryPlot,
            #            self.make_path('phase-space-z.png', i),
            #            "Macro trajectory in latent phase space",
            #            zs, titles, zlabels)
            raw = self.raw_macro
            if raw is not None and 4 <= raw.ndim <= 5 and raw.shape[-1] == 2:
                if raw.ndim == 4:
                    # Assume a deterministic network with following raw shape:
                    # (simulation, timestep, ensemble, latent state).
                    mu = raw[i, :, :, :]
                    sigma2 = 0.0 * mu
                else:
                    # Assume a probabilistic network with following raw shape:
                    # (simulation, timestep, ensemble, latent state, mu/sigma2).
                    mu = raw[i, :, :, :, 0]
                    sigma2 = raw[i, :, :, :, 1]

                kwargs = dict(
                        report=self.args.traj_report,
                        ylim=self.args.traj_ylim,
                        z_labels=self.z_labels,
                        z_shift=self.args.traj_z_shift,
                        scale_std=self.args.traj_scale_std)
                yield Task(plot_probabilistic_rnn_ensemble_t_mu_sigma,
                           self.make_path('t-z-ensemble.png', i),
                           "Probabilistic RNN ensemble outputs: "
                           "latent state mu and sigma (individual)",
                           "RNN #{j}", timestep, mu, sigma2, uncertainty, **kwargs)
                mu, sigma2 = adaled.merge_ensemble_of_mu_sigma2(
                        mu, sigma2, True, axis=1)
                yield Task(plot_probabilistic_rnn_ensemble_t_mu_sigma,
                           self.make_path('t-z-ensemble-mean.png', i),
                           "Probabilistic RNN ensemble outputs: "
                           "latent state mu and sigma (merged)",
                           None, timestep, mu[:, None], sigma2[:, None],
                           uncertainty, **kwargs)

    def tasks_spatial_channels(self):
        """Plots of u(x, t) data where u is a small vector."""
        if not _is_array(self.x_micro):
            self.skip("x is not an array")
        if self.x_micro.ndim != 4:
            self.skip(f"x is not (trajectory in batch, time step, channel, length), got {self.x_micro.shape}")
        if self.x_micro.shape[2] > self.MAX_CHANNELS:
            self.skip("too many channels")

        xlabels = self.context.xlabels
        for i, (x_micro, x_macro) in enumerate(zip(self.x_micro, self.x_macro)):
            x_micro = np.moveaxis(x_micro, 1, 0)
            x_macro = np.moveaxis(x_macro, 1, 0)
            # TODO: add stage somehow to the plot
            yield Task(plots.ImagePlotMatrix, self.make_path('uxt.png', i),
                       "AdaLED window: real trajectory",
                       [x_micro], [xlabels])
            labels = [(l, l + " (reconstructed)", l + " (absolute error)")
                      for l in xlabels]
            yield Task(plots.ImageErrorPlotMatrix,
                       self.make_path('uxt-reconstructed.png', i),
                       "AdaLED window: real trajectory, reconstructed trajectory and error",
                       list(zip(x_micro, x_macro)), labels)


def _process_config(record: adaled.TensorCollection, config):
    if config.num_trajectories > 0:
        record.describe()
        record = slice_record_trajectory_batch(
                record, slice(0, config.num_trajectories))

    version = record['version']
    fields = record['fields']
    trajectory_length = len(fields['metadata'])
    if config.slice:
        fields = slice_record_fields(fields, slice(*config.slice))

    output_prefix = os.path.splitext(config.path)[0] + '-'
    return (version, fields, output_prefix)


if __name__ == '__main__':
    TrajectoryPlotter.main()
