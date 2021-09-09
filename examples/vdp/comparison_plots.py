#!/usr/bin/env python3

import adaled
import adaled.plotting
from adaled.plotting.comparison_plots import ComparisonPlotter, Stats
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.plotting.utils import \
        fake_log_yscale, fake_log_yscale_violinplot, get_default_rgba
from examples.vdp.setup import Config

import numpy as np

from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
import argparse
import os

def _log_log_fit(x: np.ndarray, y: np.ndarray):
    log_x = np.log(x)
    poly, var = np.polyfit(log_x, np.log(y), 1, cov=True)
    print(f"LOG LOG FIT: poly={poly}  var={np.diag(var) ** 0.5}")
    fit_log_y = np.poly1d(poly)(log_x)
    return poly, var, fit_log_y


@dataclass
class VdPStats(Stats):
    config: Config  # Override.
    post: adaled.TensorCollection
    accepted_val_error_final: np.ndarray  # Final macro step validation error.
    accepted_val_error_mean: np.ndarray   # Mean macro validation error.


class VdPComparisonPlotter(ComparisonPlotter):
    def __init__(self, *args, xlabel_key: str, stats_cls=VdPStats, **kwargs):
        super().__init__(*args, stats_cls=stats_cls, **kwargs)

        if xlabel_key == 'uncertainty':
            unc = np.array([stat.config.criteria.max_uncertainty for stat in self.stats])
            self.x_values = (2 * unc) ** 0.5  # sigma_adaled^2 -> sigma_report
            self.xlabel_var = "\sigma_{\mathrm{max}}"
            self.xlabel = "uncertainty threshold $\sigma_{\mathrm{max}}$"
            self.xticklabels = [f'{sigma:.4f}' for sigma in self.x_values]
            self.fit_slice = slice(6, None)
        elif xlabel_key == 'ensemble_size':
            self.x_values = [stat.config.ensemble_size for stat in self.stats]
            self.xlabel_var = "K"
            self.xlabel = "ensemble size $K$"
            self.xticklabels = [str(ensemble_size) for ensemble_size in self.x_values]
            self.xticklabels_kwargs = {}  # No rotation.
            self.fit_slice = slice(None)
        else:
            raise ValueError(xlabel_key)

    def compute_stats_impl(self, dir: str) -> Dict[str, Any]:
        try:
            post = adaled.load(os.path.join(dir, 'postprocessed-macro-cycles.pt'))
        except FileNotFoundError:
            raise Exception(f"first run ./postprocess.py for {dir!r}")
        if post['version'] != 3:
            raise RuntimeError("wrong version, rerun ./postprocess.py")
        accepted = post['accepted_cycles_info']

        out = super().compute_stats_impl(dir)
        out['post'] = post
        out['accepted_val_error_final'] = accepted['final_validation_error']
        out['accepted_val_error_mean'] = accepted['mean_validation_error']
        macro_util = out['macro_util']
        print(f"{dir}  util={macro_util.min():.5f}..{macro_util.max():.5f}  "
              f"{macro_util.mean():.5f}+-{macro_util.std():.5f}  "
              f"error={accepted['mean_validation_error'].mean():.5f}")
        return out

    def plot_utilization(self, *args, ylim=None, **kwargs):
        if ylim is None:
            config: Config
            if max(stat.config.criteria.expected_max_utilization() for stat in self.stats) <= 0.8:
                ylim = (0.0, 0.8)
            else:
                ylim = (0.0, 1.0)

        return super().plot_utilization(*args, ylim=ylim, **kwargs)

    def plot_validation_error(self, path: str, final: bool, legend_loc: str):
        plot = Plot(path, figsize=(self.FIGURE_WIDTH, 2.5))
        ax = plot.ax

        # Violin plot works in linear space, we have to manually handle the log scale.
        if final:
            errors = [stat.accepted_val_error_final for stat in self.stats]
        else:
            errors = [stat.accepted_val_error_mean for stat in self.stats]
        mean_errors = np.array([err.mean() for err in errors])

        x = np.arange(len(self.x_values))

        # We must manually emulate log scale.
        fake_log_yscale_violinplot(ax, errors, x, widths=0.9, showmeans=True)

        poly, var, fit_log_y = _log_log_fit(self.x_values[self.fit_slice], mean_errors[self.fit_slice])
        exp = f"{poly[0]:.2f} \pm {var[0, 0] ** 0.5:.2f}"
        ax.plot(x[self.fit_slice], np.log10(np.exp(fit_log_y)),
                label="${\sim}%s^{%s}$" % (self.xlabel_var, exp))

        ax.set_xlabel(self.xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, **self.xticklabels_kwargs)
        if final:
            ax.set_ylabel(r"final macro-only error $E^c_{\mathrm{final}}$")
            ax.set_ylim(-4.0, np.log10(2) + 1e-6)  # Log.
        else:
            ax.set_ylabel("mean macro-only error $E^c_{\mathrm{mean}}$")
            ax.set_ylim(-4.0, np.log10(2) + 1e-6)  # Log.
        fake_log_yscale(ax)
        ax.grid()
        ax.legend(loc=legend_loc)
        return plot


def create_parser():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--prefix', type=str, default='', help="output file path prefix")
    add('--suffix', type=str, default='comparison', help="output file path suffix")
    add('--xlabel-key', type=str, default='uncertainty',
        choices=('uncertainty', 'ensemble_size'), help="what is being compared")
    add('--legend-loc', type=str, default='upper right', help="legend location")
    add('dirs', type=str, nargs=argparse.REMAINDER, help="runs to compare")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    plotter = VdPComparisonPlotter(args.dirs, xlabel_key=args.xlabel_key)
    plotter.plot_utilization(
            f'cmp-{args.prefix}-utilization-{args.suffix}.pdf').finalize()
    plotter.plot_validation_error(
            f'cmp-{args.prefix}-validation-error-final-{args.suffix}.pdf',
            final=True, legend_loc=args.legend_loc).finalize()
    plotter.plot_validation_error(
            f'cmp-{args.prefix}-validation-error-mean-{args.suffix}.pdf',
            final=False, legend_loc=args.legend_loc).finalize()


if __name__ == '__main__':
    main()
