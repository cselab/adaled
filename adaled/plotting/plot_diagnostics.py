#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adaled import AdaLEDStage, TensorCollection
from adaled.led.diagnostics import \
        AdaLEDDiagnostics, HistogramStatsConfig
from adaled.plotting.base import Plotter, Task
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.postprocessing.record import compute_macro_utilization
from adaled.utils.arrays import rolling_average
from adaled.utils.glob_ import glob_ex
import adaled

import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Union
import argparse
import warnings


# https://matplotlib.org/stable/gallery/color/named_colors.html
LOSSES_DEFAULT = {
    'transformer_train': ('-', None, 'green', "AE training loss"),
    'transformer_valid': ('-', None, 'limegreen', "AE validation loss"),
    'macro_train': ('-', None, 'darkblue', "RNN training loss"),
    'macro_train.TOTAL': ('-', 2.0, 'mediumblue', "RNN training total loss"),
    'macro_train.adversarial': (':', 1.0, 'darkblue', "RNN training adversarial loss"),
    'macro_train.mse': ('-.', 1.0, 'darkblue', "RNN training MSE loss"),
    'macro_train.nll': ('--', 1.0, 'darkblue', "RNN training NLL loss"),
    'macro_train.original.mse': ('-.', 1.0, 'darkblue', "RNN training MSE loss"),
    'macro_train.original.nll': ('--', 1.0, 'darkblue', "RNN training NLL loss"),
    'macro_valid': ('-', None, 'dodgerblue', "RNN validation loss"),
    'macro_valid.TOTAL': ('-', 2.0, 'deepskyblue', "RNN validation total loss"),
    'macro_valid.mse': ('-.', 1.0, 'dodgerblue', "RNN validation MSE loss"),
    'macro_valid.nll': ('--', 1.0, 'dodgerblue', "RNN validation NLL loss"),
}

def _get_cyclic(array, i):
    return array[i % len(array)]

def _normalize_losses(losses: Union[Sequence[float], TensorCollection]):
    """Flatten the structure and convert to numpy. Compute total loss for each
    of the main 4 categories (transformer and macro training and validation
    loss).

    In case of the old data format where 4 losses were hardcoded, convert to
    TensorCollection first.
    """
    losses = losses.cpu_numpy()

    out = {}
    for key, value in losses.items():
        if isinstance(value, TensorCollection):
            # Concat each part separately to control the order of lines.
            out[key + '.TOTAL'] = sum(value.allvalues())
            for k, v in value.concat_flatten().items():
                out[key + '.' + k] = v
        else:
            out[key] = value

    return TensorCollection(out)


def _plot_common(ax: Axes, x: Sequence[int], xlabel: str):
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, x[-1])
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_ticks_position('both')
    ax.grid()
    ax.legend()


def _plot_stats1(path: str,
                 stats_list: List[TensorCollection],
                 x: Sequence[int],
                 xlabel: str,
                 losses_kwargs: Dict[str, Dict[str, Any]]):
    tmp = losses_kwargs
    stats = stats_list[0]
    if len(stats_list) > 1:
        warnings.warn("plotting only 0th rank's diagnostics, multi-rank plots not implemented")
    losses = _normalize_losses(stats['losses'])
    with Plot(path) as (fig, ax):
        ax: Axes
        for key, loss in losses.items():
            if '.TOTAL' in key or '.nll' in key or 'adversarial' in key:
                continue
            mask = ~np.isnan(loss)
            if mask.any():
                loss = loss[mask]
                if loss.any():
                    ax.plot(x[mask], loss, **losses_kwargs[key])
        error = stats['cmp_error']
        uncertainty = stats['cmp_uncertainty']
        validation_error = stats.get('validation_error')
        # Is this the new format? (old already stored mean error)
        # DEPRECATED (2021-11-10): always assume batch format.
        if error.ndim > 1:
            error = error.mean(axis=tuple(range(1, error.ndim)))
            uncertainty = uncertainty.mean(axis=tuple(range(1, uncertainty.ndim)))
            if validation_error is not None:
                validation_error = validation_error.mean(
                        axis=tuple(range(1, validation_error.ndim)))
        macro_steps = np.float32(stats['macro_steps'])
        macro_steps[macro_steps == 0] = np.nan
        # TODO: Add shaded region for std dev (or min/max) of error and
        # uncertainty, since we have a batch of simulations?
        ax.plot(x, error, color='darkred', label="macro-micro error")
        ax.plot(x, uncertainty, color='orange', label="macro uncertainty")
        ax.plot(x, macro_steps, color='black', label="number of macro steps", marker='o')
        if validation_error is not None:
            ax.plot(x, validation_error, color='red', label="validation macro-micro error")
        ax.set_title("AdaLED diagnostics")
        _plot_common(ax, x, xlabel)


def _plot_stats2(path: str,
                 stats_list: List[TensorCollection],
                 x: Sequence[int],
                 xlabel: str):
    sizes_list = [stats.get('dataset') for stats in stats_list]
    if any(sizes is None for sizes in sizes_list):
        print(f"old diagnostics format, skipping {path}")
        return
    stats = stats_list[0]
    sizes = sizes_list[0]
    if len(stats_list) > 1:
        warnings.warn("plotting only 0th rank's diagnostics, multi-rank plots not implemented")
    with Plot(path) as (fig, ax):
        ax: Axes
        train = sizes['train']
        valid = sizes['valid']
        # Plots are ordered in the decreasing order, such that their order in
        # the label and figure match.
        ax.plot(x, train['num_states'], color='black', linestyle='-',
                label="# of training states")
        ax.plot(x, valid['num_states'], color='black', linestyle='--',
                label="# of validation states")
        ax.plot(x, train['num_trajectories'], color='black', linestyle=':',
                label="# of training trajectories")
        ax.plot(x, valid['num_trajectories'], color='black', linestyle='-.',
                label="# of validation trajectories")

        dt = np.diff(stats['end_wall_time'])
        # Note: completely wrong when server and client are on seperate ranks.
        ax.plot(x[1:], dt, color='orange', label="cycle execution time? [s]")

        hyperparams = stats['trainer_hyperparams']

        def _plot_lr(where, label, factor, **kwargs):
            try:
                lr = hyperparams[where]['lr']
            except KeyError:
                return

            if lr.ndim == 1:
                lr = lr.reshape(len(lr), 1)
            else:
                lr = lr.reshape(len(lr), -1)
            # Artificial factor to split the overlapping lines.
            lr *= factor ** np.arange(lr.shape[-1])
            ax.plot(x, lr[:, 0], label=label, **kwargs)
            for lr_ in lr.T:
                ax.plot(x, lr_, **kwargs)

        _plot_lr('transformer', "transformer LR", 1.0, color='green')
        _plot_lr('macro', "macro LR", 1.1, color='blue', lw=0.8)

        ax.set_title("AdaLED diagnostics (cont.)")
        _plot_common(ax, x, xlabel)


def _plot_stats3(
        path: str,
        stats_list: List[TensorCollection],
        x: Sequence[int],
        xlabel: str):
    """Plot macro steps only, in a linear plot."""
    if len(stats_list) > 1:
        warnings.warn("plotting only 0th rank's diagnostics, multi-rank plots not implemented")
    stats = stats_list[0]
    with Plot(path) as (fig, ax):
        ax: Axes
        ax.stairs(stats['macro_steps'][:-1], x, color='black', fill=True,
                label="number of macro steps")
        ax.grid()
        ax.legend()
        ax.set_title("AdaLED diagnostics: number of macro steps")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, x[-1])


def _plot_losses(path: str,
                 stats_list: List[TensorCollection],
                 x: Sequence[int],
                 xlabel: str,
                 yscale: str,
                 losses_kwargs: Dict[str, Dict[str, Any]]):
    stats = stats_list[0]
    if len(stats_list) > 1:
        warnings.warn("plotting only 0th rank's diagnostics, multi-rank plots not implemented")
    losses = _normalize_losses(stats['losses'])
    with Plot(path) as (fig, ax):
        ax: mpl.axes.Axes
        for key, loss in losses.items():
            mask = ~np.isnan(loss)
            if mask.any():
                ax.plot(x[mask], loss[mask], **losses_kwargs[key])
        # ax.plot(x, stats['cmp_mse'], color='darkred', label="macro-micro MSE")
        # ax.plot(x, stats['cmp_uncertainty'], color='orange', label="macro uncertainty")
        ax.set_title("AdaLED Losses")
        ax.set_xlabel(xlabel)
        if yscale == 'symlog':
            ax.set_yscale(yscale, linthresh=1e-3)
        else:
            ax.set_yscale(yscale)
        ax.set_xlim(0, x[-1])
        # Minor ticks for some reason don't show up in the symlog plot, the
        # only one where they would be helpful.
        # ax.minorticks_on()
        # ax.tick_params(axis='x', which='minor', bottom=False)
        ax.grid()
        ax.legend()


class DiagnosticsAnimationPlot(Plot):
    TRANSFORMER_COLORS = ['black']
    TRANSFORMER_LINE_STYLES = ['--', '-.', ':']
    def __init__(self,
                 path_or_axes,
                 fields: TensorCollection,
                 diagnostics: TensorCollection,
                 *,
                 error_label: str,
                 F_label: str,
                 macro_roll_window: int = 1000,
                 criteria: Optional[adaled.SimpleCriteriaConfig] = None,
                 legend: bool = True,
                 **kwargs):
        super().__init__(path_or_axes, **kwargs)
        sim = fields['simulations']
        t = fields['metadata', 'timestep']
        F = sim['F'] if 'F' in sim else np.full(len(t), np.nan)
        cycle_t = diagnostics['start_timestep']

        ax: Axes = self.ax
        ax_macro: Axes = ax.twinx()
        ax_F: Axes = ax.twinx()
        self.ax_macro = ax_macro
        self.ax_F = ax_F
        ax.set_zorder(3)
        ax_F.set_zorder(2)
        ax_macro.set_zorder(1)
        ax.patch.set_visible(False)
        ax_F.patch.set_visible(False)

        # Comparsion MSE + uncertainty and timestep uncertainty.
        if criteria:
            ax.axhline(criteria.max_cmp_error, color='red', linewidth=1.5)
            ax.axhline(criteria.max_uncertainty, color='orange', linewidth=1.5)
        cmp_error = diagnostics['cmp_error'][:, 0]  # Only the first trajectory.
        cmp_uncertainty = diagnostics['cmp_uncertainty'][:, 0]
        # User can delete uncertainty to skip plotting it.
        uncertainty = sim.get('uncertainty')
        cmp_error_line, = ax.plot(cycle_t, cmp_error, color='red',
                              label=error_label, linewidth=1.2)
        cmp_uncertainty_line, = ax.plot(
                cycle_t, cmp_uncertainty, color='orange', linestyle='--',
                label="uncertainty", linewidth=1.2)
        if uncertainty is not None:
            # Use same color and style for now as for the cmp uncertainty.
            uncertainty_line, = ax.plot(t, uncertainty, color='orange', linewidth=1.2)

        # Macro and transformer losses.
        macro_loss = diagnostics['losses'].get('macro_train')
        if isinstance(macro_loss, TensorCollection):
            # TODO: Make this generic.
            try:
                macro_loss = macro_loss['original', 'mse']
            except KeyError:
                macro_loss = macro_loss['mse']
        elif macro_loss is not None and macro_loss.ndim != 1:
            raise TypeError(f"unrecognized training loss shape: {macro_loss.shape}")
        if macro_loss is not None:
            macro_loss_line, = ax.plot(cycle_t, macro_loss, color='#333333',
                                       label="macro t.l.", linewidth=0.8)
        transformer_loss = diagnostics['losses'].get('transformer_train')
        if transformer_loss is not None:
            if isinstance(transformer_loss, TensorCollection):
                transformer_loss = transformer_loss.concat_flatten('/')
            else:
                transformer_loss = {'': transformer_loss}

            # Filter out values that are all nans or zeros.
            transformer_loss = {k: v for k, v in transformer_loss.items()
                                if np.nanmin(v) < np.nanmax(v)}
            transformer_loss_lines = [
                ax.plot(cycle_t, loss,
                        color=_get_cyclic(self.TRANSFORMER_COLORS, i),
                        linestyle=_get_cyclic(self.TRANSFORMER_LINE_STYLES, i),
                        label=f"transformer t.l. {key}", linewidth=0.8)[0]
                for i, (key, loss) in enumerate(transformer_loss.items())
            ]

        # Rolling average, fraction of macro steps.
        stages = fields['metadata', 'stage']
        # _is_macro = stages == int(AdaLEDStage.MACRO)
        _cycle_t, cycle_utilization = compute_macro_utilization(t, stages)
        # fm_steps = rolling_average(_is_macro, macro_roll_window)
        # fm_line, = ax_macro.plot(t, fm_steps, color='green', linewidth=1.0,
        #                          label="fraction of macro steps")
        util_color = 'green'
        util_stairs = ax_macro.stairs(cycle_utilization, _cycle_t,
                                      color=util_color, alpha=0.3, fill=True)
        F_lines = ax_F.plot(t, F, color='blue', label=F_label, linewidth=0.8)

        if legend:
            # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_macro.get_legend_handles_labels()
            lines3, labels3 = ax_F.get_legend_handles_labels()
            ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                      loc='upper left', labelspacing=0.1, fontsize='small')
        ax.grid(color='lightgray')
        ax.set_xlabel("time step")
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(1e-6, 1e2)
        ax.set_ylabel(f"training loss and {error_label}")
        ax.set_yscale('log')
        ax_macro.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax_macro.set_ylim(0.0, 0.8)  # 80%
        ax_macro.set_ylabel("fraction of macro steps", color=util_color)
        ax_macro.spines['right'].set_edgecolor(util_color)
        ax_macro.tick_params(axis='y', colors=util_color)
        ax_F.set_ylabel(F_label, color=F_lines[0].get_color())
        ax_F.spines['right'].set_edgecolor(F_lines[0].get_color())
        ax_F.spines['right'].set_position(('outward', 60))
        ax_F.tick_params(axis='y', colors=F_lines[0].get_color())

        def update_func(frame: int, cycle: int, last_cycle: int):
            updated = [util_stairs, *F_lines]
            if F.ndim == 1:
                assert len(F_lines) == 1
                F_lines[0].set_data(t[:frame], F[:frame])
            elif F.ndim == 2:
                for i in range(len(F_lines)):
                    F_lines[i].set_data(t[:frame], F[:frame, i])
            else:
                raise NotImplementedError("only 0-dimensional or 1-dimensional F supported")

            if cycle != last_cycle:
                util_stairs.set_data(
                        cycle_utilization[:cycle],
                        cycle_t[:min(len(cycle_utilization), cycle) + 1])
                updated.append(cmp_error_line)
                updated.append(cmp_uncertainty_line)
                cmp_error_line.set_data(cycle_t[:cycle], cmp_error[:cycle])
                cmp_uncertainty_line.set_data(cycle_t[:cycle], cmp_uncertainty[:cycle])
                if uncertainty is not None:
                    updated.append(uncertainty_line)
                    uncertainty_line.set_data(t[:frame], uncertainty[:frame])

                if macro_loss is not None:
                    updated.append(macro_loss_line)
                    macro_loss_line.set_data(cycle_t[:cycle], macro_loss[:cycle])
                if transformer_loss is not None:
                    updated.extend(transformer_loss_lines)
                    for line, loss in zip(transformer_loss_lines, transformer_loss.values()):
                        line.set_data(cycle_t[:cycle], loss[:cycle])
            return updated

        self.update = update_func


class HistogramAnimationPlot(Plot):
    def __init__(self,
                 path_or_axes,
                 histogram_sets: Sequence[Sequence[np.ndarray]],
                 config_sets: Sequence[Sequence[HistogramStatsConfig]],
                 label_sets: Sequence[Sequence[Optional[str]]],
                 xlabels: Sequence[Optional[str]],
                 *,
                 ylabel: str,
                 **kwargs):
        super().__init__(path_or_axes, **kwargs)

        # https://matplotlib.org/stable/gallery/animation/animated_histogram.html
        containers = []
        histos = []
        for ax, histograms, configs, labels, xlabel in \
                zip(self.ax, histogram_sets, config_sets, label_sets, xlabels):
            ax: Axes
            for h, c, l in zip(histograms, configs, labels):
                histos.append(h)
                containers.append(ax.stairs(h[0], c.get_bin_edges(), label=l))
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter(sep=""))
            ax.set_xlabel(xlabel)
            log = all(c.log for c in configs)
            assert all(log == c.log for c in configs)
            if log:
                ax.set_xscale('log')
            ax.set_ylim(0, max(h.max() for h in histograms))
            ax.set_ylabel(ylabel)
            if any(labels):
                ax.legend()

        def update_func(frame: int, cycle: int, last_cycle: int):
            if cycle == last_cycle:
                return []
            updated = []
            for container, histogram in zip(containers, histos):
                container.set_data(histogram[cycle])
                updated.append(container)
            return updated

        self.update = update_func


class DiagnosticsPlotter(Plotter):
    LOSSES = LOSSES_DEFAULT

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        add = parser.add_argument_group('data').add_argument
        # diagnostics.pkl is a legacy (2021-11-19)
        add('--diagnostics', type=str, nargs='+',
            default=['diagnostics*.pkl', 'diagnostics*.pt'],
            help="paths of the diagnostics dump files")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diagnostics = None

    def set_up(self):
        if self._diagnostics is None:
            paths = glob_ex(self.args.diagnostics)
            self._diagnostics = [AdaLEDDiagnostics.load(path) for path in paths]
            self._diagnostics = \
                    [d for d in self._diagnostics if len(d.per_cycle_stats.data) > 0]
        if not self._diagnostics:
            self.skip("empty diagnostics")
        return self._diagnostics

    def make_path(self, path):
        return path

    def losses_kwargs(self):
        losses_kwargs = {
            key: dict(zip(['linestyle', 'linewidth', 'color', 'label'], args))
            for key, args in self.LOSSES.items()
        }
        return losses_kwargs

    def tasks_stats(self):
        stats_list = [diag.per_cycle_stats.data for diag in self._diagnostics]
        first = stats_list[0]
        x_cycles = np.arange(1, 1 + len(first))  # First cycle is skipped.
        x_timesteps = first['start_timestep']

        plot_fns = [_plot_stats1, _plot_stats2, _plot_stats3]
        for i, fn in enumerate(plot_fns, 1):
            kwargs = {'losses_kwargs': self.losses_kwargs()} if i == 1 else {}
            yield Task(fn, self.make_path(f'stats{i}-vs-cycle.png'),
                       stats_list, x=x_cycles, xlabel="AdaLED cycle", **kwargs)
            yield Task(fn, self.make_path(f'stats{i}-vs-timestep.png'),
                       stats_list, x=x_timesteps, xlabel="time step", **kwargs)
            if 'total_train_samples' in first:
                yield Task(fn, self.make_path(f'stats{i}-vs-train-samples.png'),
                           stats_list,
                           x=first['total_train_samples', 'macro_train'],
                           xlabel="total samples trained on", **kwargs)

    def tasks_losses(self):
        stats_list = [diag.per_cycle_stats.data for diag in self._diagnostics]
        first = stats_list[0]
        x_cycles = np.arange(1, 1 + len(first))  # First cycle is skipped.
        yield Task(_plot_losses, self.make_path('losses-vs-cycle-linear.png'),
                   stats_list, x_cycles, "AdaLED cycle", yscale='linear',
                   losses_kwargs=self.losses_kwargs())
        yield Task(_plot_losses, self.make_path('losses-vs-cycle-symlog.png'),
                   stats_list, x_cycles, "AdaLED cycle", yscale='symlog',
                   losses_kwargs=self.losses_kwargs())


if __name__ == '__main__':
    DiagnosticsPlotter.main()
