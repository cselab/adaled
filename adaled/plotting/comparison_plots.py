from adaled.led import AdaLEDStage as Stage
from adaled.utils.arrays import masked_gaussian_filter1d
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.plotting.utils import darken_color, get_default_rgba
import adaled

import numpy as np

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import dataclasses
import os
import warnings

FIGWIDTH = float(os.environ.get('ADALED_FIGWIDTH', 8.0))

# TODO: The plots below use stats.postprocessed and cyl-specific namings.
#       Plots can be reused, but the variable names should be formalized.

@dataclass
class Stats:
    dir: str
    config: Any
    diag: adaled.AdaLEDDiagnostics
    macro_util: np.ndarray  # Macro utilization.


@dataclass
class StatsGroup:
    key: Any
    indices: List[int]
    stats: List[Stats]


def config_as_key(stat: Stats) -> str:
    key = repr(dataclasses.asdict(stat.config))
    return key


class ComparisonPlotter:
    FIGURE_WIDTH = float(os.environ.get('ADALED_FIGURE_WIDTH', 8.0))

    def __init__(self, dirs: Sequence[str], stats_cls: type = Stats):
        self.stats_cls = stats_cls
        self.stats = [self.compute_stats(dir) for dir in dirs]
        self.stats_groups = self.group_stats(self.stats)

        self.xlabel = "x"
        self.xticklabels = dirs
        self.xticklabels_kwargs = {'rotation': 45, 'ha': 'right'}

    def group_stats(
            self,
            stats: List[Stats],
            key_func: Callable[[Stats], Any] = config_as_key) \
                    -> Dict[Any, StatsGroup]:
        groups: Dict[Any, StatsGroup] = {}

        for i, stat in enumerate(stats):
            key = key_func(stat)
            if key in groups:
                group = groups[key]
            else:
                groups[key] = group = StatsGroup(key=key, indices=[], stats=[])
            group.indices.append(i)
            group.stats.append(stat)

        return groups

    def compute_stats_impl(self, dir: str) -> Dict[str, Any]:
        config = adaled.load(os.path.join(dir, 'config.pt'))
        diag = adaled.AdaLEDDiagnostics.load(os.path.join(dir, 'diagnostics-000.pt'))
        data = diag.per_cycle_stats

        start = data['start_timestep']

        # Note: no idea why :-1 gives correct result, logically it should be 1:.
        macro_util = data['macro_steps', :-1] / (start[1:] - start[:-1])

        return dict(dir=dir, config=config, diag=diag, macro_util=macro_util)

    def compute_stats(self, dir: str) -> Stats:
        return self.stats_cls(**self.compute_stats_impl(dir))

    def plot_execution_time(self, path: str):
        x = np.arange(len(self.stats))
        ts = []
        with Plot(path) as (fig, ax):
            for stat in self.stats:
                ta = stat.postprocessed['runtime', 'metadata', 'execution_time']
                tb = stat.diag.per_cycle_stats['stats_overhead_seconds']
                t = ta[1:].sum() - tb.sum()
                ts.append(t)

            ax.plot(x, ts)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("execution time [s]")
            ax.grid()

    def plot_utilization(
            self,
            path: str,
            ylim: Tuple[float, float] = (0.0, 1.0)) -> Plot:
        plot = Plot(path, figsize=(self.FIGURE_WIDTH, 2.5))
        ax = plot.ax

        x = np.arange(len(self.stats))
        util = [stat.macro_util for stat in self.stats]

        violin = ax.violinplot(util, x, widths=0.9, showextrema=False, showmeans=True)
        for pc in violin['bodies']:
            pc.set_facecolor('green')
        violin['cmeans'].set_edgecolor('green')

        ax.set_xlabel(self.xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, **self.xticklabels_kwargs)
        ax.set_ylabel("macro utilization", color='darkgreen')
        ax.set_ylim(*ylim)
        ax.spines['left'].set_edgecolor('darkgreen')
        ax.tick_params(axis='y', colors='darkgreen')
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.grid()
        return plot

    def plot_F_vs_t_grouped(
            self,
            path: str,
            F_title: str = "F",
            stride: int = 50) -> Plot:
        """Plot multiple F profiles. Skip duplicates."""
        plot = Plot(path, figsize=(FIGWIDTH, 1.5))
        ax: Axes = plot.ax

        Fcolor = 'blue'
        Fcolor_dark = mpl.colors.to_hex(darken_color(Fcolor, 0.2))

        groups = self.stats_groups
        prev_F = None
        for group, color in zip(groups.values(), get_default_rgba(len(groups))):
            t = group.stats[0].postprocessed['runtime', 'metadata', 'timestep'][::stride]
            F = group.stats[0].postprocessed['runtime', 'F'][::stride]
            if prev_F is None or np.abs(F - prev_F).max() > 1e-6:
                ax.plot(t, F, linewidth=0.7, color=Fcolor)
                prev_F = F

        ax.set_xlim(t.min(), round(t.max(), -2))
        ax.set_ylabel(F_title, color=Fcolor_dark)
        ax.tick_params(axis='y', colors=Fcolor_dark)
        ax.spines['left'].set_edgecolor(Fcolor_dark)
        ax.grid(color='lightgray')

        return plot

    def plot_field_vs_t_grouped(
            self,
            path: str,
            multikey: Tuple[str, ...] = ('runtime', 'metadata', '__is_macro'),
            ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
            ylabel: Optional[str] = None,
            yscale: str = 'linear',
            ycolor: Optional[str] = None,
            yplot_kwargs: Dict[str, Any] = {},
            stride: int = 50,
            smoothing_before: float = 500.0,
            smoothing_after: float = 0.0) -> Plot:
        """
        Args:
            ylim: defaults to (0.0, 1.0) if __is_macro is plotted
            ylabel: defaults to "macro utilization" if __is_macro is plotted
        """
        if multikey == ('runtime', 'metadata', '__is_macro'):
            if ylim is None:
                ylim = (0.0, 1.0)
            if ylabel is None:
                ylabel = "macro utilization"
            if ycolor is None:
                ycolor = 'darkgreen'
        if ycolor is None:
            ycolor = 'black'

        plot = Plot(path, figsize=(FIGWIDTH, 1.5))
        ax: Axes = plot.ax

        groups = self.stats_groups
        for group, color in zip(groups.values(), get_default_rgba(len(groups))):
            print(f"Processing group containing {group.stats[0].dir}...")
            fields = []
            raw = []
            for stat in group.stats:
                field = stat.postprocessed[multikey]
                raw.append(field)
                if smoothing_before:
                    field = masked_gaussian_filter1d(
                            field.astype(np.float32), smoothing_before, set_nan=True)
                fields.append(field)
            fields = np.stack(fields)
            raw = np.stack(raw)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(fields, axis=0)
                if len(group.stats) > 1:
                    std = np.nanstd(fields, axis=0)
            if smoothing_after:
                mean = masked_gaussian_filter1d(mean, smoothing_after, set_nan=True)
            mean = mean[::stride]

            t = group.stats[0].postprocessed['runtime', 'metadata', 'timestep'][::stride]
            if len(group.stats) > 1:
                if smoothing_after:
                    std = masked_gaussian_filter1d(std, smoothing_after, set_nan=True)
                std = std[::stride]
                ax.fill_between(t, mean - std, mean + std, facecolor=color, alpha=0.2)
            else:
                std = ["N/A"] * len(t)

            label = self.xticklabels[group.indices[0]]
            for _ in [len(t) // 2, len(t) - 1]:
                print(f"{label} t={t[_]}  [{multikey}]={mean[_]}+-{std[_]}")

            raw_mean = np.nanmean(raw, axis=0)
            for _ in [slice(0, len(raw_mean) // 2), slice(len(raw_mean) // 2, None)]:
                part = raw_mean[_]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    print(f"{label} t={_}  [{multikey}]={np.nanmean(part)}+-{np.nanstd(part)}")
            ax.plot(t, mean, **yplot_kwargs, color=color, label=label)

        if ylabel:
            ax.set_ylabel(ylabel, color=ycolor)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_yscale(yscale)
        ax.set_xlim(t.min(), round(t.max(), -2))
        ax.spines['left'].set_edgecolor(ycolor)
        ax.tick_params(axis='y', colors=ycolor)
        if multikey == ('runtime', 'metadata', '__is_macro'):
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.grid(color='lightgray')
        ax.set_axisbelow(True)

        return plot

    def plot_final_grouped(
            self,
            path: str,
            what: Callable[[Stats], float],
            ylabel: str,
            ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
            yscale: str = 'linear',
            yplot_kwargs: Dict[str, Any] = {},
            sem: bool = False) -> Plot:
        """
        Plot accumulated values, one per group.

        Args:
            what: callable that returns a value, given a Stats instance
            sem: whether to plot the standard error of mean
        """
        if sem:
            import scipy.stats

        plot = Plot(path, figsize=(8.0, 2.5))
        ax = plot.ax

        groups = self.stats_groups
        means = []
        stds = []
        sems = []
        for group, color in zip(groups.values(), get_default_rgba(len(groups))):
            print("GROUP", len(group.stats))
            values = []
            for stat in group.stats:
                value = what(stat)
                if not isinstance(value, (int, float, np.generic)):
                    raise Exception(f"expected a number, got {value}")
                values.append(value)
            values = np.array(values)
            means.append(values.mean())
            if len(group.stats) > 1:
                stds.append(values.std())
                if sem:
                    sems.append(scipy.stats.sem(values))

        color = get_default_rgba(1)[0]
        means = np.array(means)
        x = np.arange(len(means))
        if stds:
            stds = np.array(stds)
            ax.fill_between(x, means - stds, means + stds, facecolor=color, alpha=0.2)
        if sem:
            sems = np.array(sems)
            ax.plot(x, means - sems, color=color, linestyle='--')
            ax.plot(x, means + sems, color=color, linestyle='--')
        ax.plot(x, means, color=color)

        xlabels = [self.xticklabels[group.indices[0]] for group in groups.values()]

        ax.grid()
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_xticks(x)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_yscale(yscale)
        return plot
