#!/usr/bin/env python3

import cubismup2d as cup2d  # Import immediately.
from .setup import Config
from .postprocess_utils import load_and_extend_runtime_postprocess_data

import adaled
import adaled.plotting
from adaled.led import AdaLEDStage as Stage
from adaled.plotting.comparison_plots import ComparisonPlotter, Stats
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.plotting.utils import \
        divider_colorbar, fade_color, fake_log_yscale, \
        fake_log_yscale_violinplot, get_default_rgba
from adaled.postprocessing.misc import pareto_front
from adaled.postprocessing.record import get_cycle_slices
from adaled.utils.dataclasses_ import dataclass

import numpy as np

from typing import Any, Dict, Optional, Sequence, Tuple
import argparse
import glob
import os

MARKERS = 'ovPxDs<Xd>p^*'

@dataclass
class CylStats(Stats):
    config: Config  # Override.
    postprocessed: Optional[adaled.TensorCollection]


class CylComparisonPlotter(ComparisonPlotter):
    def __init__(self, dirs: Sequence[str], xlabel_key: str,
                 *args, stats_cls=CylStats, **kwargs):
        super().__init__(dirs, *args, stats_cls=stats_cls, **kwargs)

        xticklabels = []
        xgeoinfos = []
        for dir, stat in zip(dirs, self.stats):
            mr = stat.config.micro.make_multiresolution()
            geo = " + ".join(
                    f"{layer.downscaled_shape[1]}x{layer.downscaled_shape[0]}"
                    for layer in mr.layers)
            if 'seed' in dir:
                geo += " (seed)"
            xgeoinfos.append(geo)

        if xlabel_key.startswith('config.'):
            xlabel = xlabel_key
            for stat in self.stats:
                info = stat.config
                for key in xlabel_key[7:].split('.'):
                    info = getattr(key, info)
                xticklabels.append(str(info))
        elif xlabel_key == 'AE':
            xlabel = "autoencoder architecture"
            for stat in self.stats:
                config: Config = stat.config
                mr = config.micro.make_multiresolution()
                ae = config.autoencoder.encoders
                layers = "-".join(map(str, ae[0].conv_layers_channels))
                xticklabels.append(f"{len(ae)}x {layers}")
        elif xlabel_key == 'vorticity_weight':
            xlabel = "vorticity weight $\lambda_\omega$"
            for stat in self.stats:
                config: Config = stat.config
                xticklabels.append(f"{config.autoencoder.loss.vorticity_weight:.5f}")
        elif xlabel_key == 'derivatives_weight':
            xlabel = "derivatives weight $\lambda_{\partial}$"
            for stat in self.stats:
                config: Config = stat.config
                xticklabels.append(f"{config.autoencoder.loss.derivatives_weight:.5f}")
        elif xlabel_key == 'index':
            xlabel = "index"
            xticklabels = [str(i) for i in range(len(self.stats))]
        elif xlabel_key == 'dir':
            xlabel = ""
            xticklabels = dirs
        elif xlabel_key == 'sigma':
            xlabel = r"$\sigma_{\mathrm{max}}$"
            for stat in self.stats:
                config: Config = stat.config
                xticklabels.append(config.led.criteria.max_uncertainty)
        elif xlabel_key == 'cylinder_mask':
            xlabel = "cylinder mask"
            for stat in self.stats:
                config: Config = stat.config
                if config.micro.enforce_obstacle_velocity:
                    sdf = config.micro.enforce_obstacle_velocity_sdf / config.micro.compute_h()
                    xticklabels.append(f"$r_{{\\mathrm{{mask}}}}=r_{{\\mathrm{{cyl}}}}{sdf:+.0f}h$")
                else:
                    xticklabels.append(f"no mask")
        elif xlabel_key == 'k_warmup_cmp':
            xlabel = "$k_{\mathrm{warmup}} + k_{\mathrm{cmp}}$"
            for stat in self.stats:
                config: Config = stat.config
                criteria = config.led.criteria
                xticklabels.append(f"W{criteria.k_warmup}-C{criteria.k_cmp}")
        else:
            raise ValueError(xlabel_key)

        self.xlabel = xlabel
        self.xticklabels = xticklabels
        self.xgeoinfos = xgeoinfos

    def compute_stats_impl(self, dir: str) -> Dict[str, Any]:
        print(f"Loading stats for {dir}.")
        out = super().compute_stats_impl(dir)
        path = os.path.join(dir, 'postprocessed-000.pt')
        per_cycle = out['diag'].per_cycle_stats
        try:
            postprocessed = adaled.load(path)
        except FileNotFoundError:
            print(f"{path} not found, not plotting losses")
            postprocessed = {}

        path = os.path.join(dir, 'postprocessed-runtime-000.pt')
        try:
            postprocessed['runtime'] = \
                    load_and_extend_runtime_postprocess_data(path, out['diag'])
        except FileNotFoundError:
            print(f"{path} not found")

        out['postprocessed'] = adaled.TensorCollection(postprocessed)
        return out

    def plot_violin_force_errors(self, path: str) -> Plot:
        plot = Plot(path, figsize=(10, 4.0))
        ax = plot.ax

        x = np.arange(len(self.stats))
        errors = []
        mean_errors = []
        for stat in self.stats:
            # stage = stat.postprocessed['metadata', 'stage']
            # forces = stat.postprocessed['postprocess', 'x_forces']
            # diff = forces['micro', 'cyl_forces'] - forces['macro', 'cyl_forces']
            # error = np.abs(diff).sum(axis=-1)
            stage = stat.postprocessed['runtime', 'metadata', 'stage']
            error = stat.postprocessed['runtime', '__qoi_error_l1', 'cyl_forces', :, 0]
            error = error[stage == Stage.MACRO]
            errors.append(np.maximum(1e-5, error))

        # We must manually emulate log scale.
        fake_log_yscale_violinplot(ax, errors, x, widths=0.9, showmeans=True)

        ax.set_xlabel(self.xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, rotation=45, ha='right')
        ax.set_ylabel("forces error")
        fake_log_yscale(ax)
        ax.grid()
        return plot

    def plot_violin(
            self,
            path: str,
            multikey: Tuple[str, ...],
            ylabel: str,
            yclip: Optional[float] = None) -> Plot:
        """Make a log violin plot."""
        plot = Plot(path, figsize=(10, 4.0))
        ax = plot.ax

        x = np.arange(len(self.stats))
        values = []
        for stat in self.stats:
            value = stat.postprocessed[multikey]
            if isinstance(value, adaled.TensorCollection):
                value = sum(value.allvalues())
            value = value[~np.isnan(value)]
            value = np.clip(value, yclip, None)
            values.append(value)

        # We must manually emulate log scale.
        fake_log_yscale_violinplot(ax, values, x, widths=0.9, showmeans=True)

        ax.set_xlabel(self.xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        fake_log_yscale(ax)
        ax.grid()
        return plot

    def plot_utilization_vs_loss_double_pareto(
            self,
            path_fmt: str,
            multikey1: Tuple[str, ...],
            multikey2: Optional[Tuple[str, ...]],
            loss_label1: str,
            loss_label2: str,
            begin: int = 1,
            mark_ids: Sequence[int] = [],
            figsize: Tuple[float, float] = (7.0, 3.0)) -> None:
        """Scatter plot of average macro utilizations vs average loss, starting
        from the time step `begin` (default to 1 to skip the first `nan`)."""
        N = len(self.stats)

        unique_geoinfos = sorted(set(self.xgeoinfos))
        utilizations = np.empty(N)
        losses_macro1 = np.empty(N)
        losses_macro2 = np.empty(N)
        losses_all1 = np.empty(N)
        losses_all2 = np.empty(N)
        for i, stat in enumerate(self.stats):
            post = stat.postprocessed[begin:]
            try:
                stage = post['metadata', 'stage']
            except:
                stage = post['runtime', 'metadata', 'stage']
            loss1 = post[multikey1][:, 0]
            loss2 = post[multikey2][:, 0]

            is_macro = stage == Stage.MACRO
            utilizations[i] = is_macro.sum() / len(stage)
            losses_macro1[i] = loss1[is_macro].sum() / is_macro.sum()
            losses_macro2[i] = loss2[is_macro].sum() / is_macro.sum()
            losses_all1[i] = loss1[is_macro].sum() / len(stage)
            losses_all2[i] = loss2[is_macro].sum() / len(stage)

        def _plot_lu(ax: Axes, losses1, losses2):
            nans = np.isnan(losses1 + losses2 + utilizations)
            assert not nans[mark_ids].any(), "nan runs shouldn't be marked"
            is_marked = np.zeros(len(nans), dtype=bool)
            is_marked[mark_ids] = True

            K = len(unique_geoinfos)
            for i, geoinfo, marker, color in \
                    zip(range(K), unique_geoinfos, MARKERS, get_default_rgba(K)):
                mask = np.array([g == geoinfo for g in self.xgeoinfos])
                mask &= ~nans
                x = losses1[mask]
                y = utilizations[mask]
                pareto1 = pareto_front(-x, y)
                pareto2 = pareto_front(-losses2[mask], y)

                print(f"Samples: {len(x)}  Pareto:")
                print(np.arange(len(nans))[mask][pareto1])
                print(np.arange(len(nans))[mask][pareto2])
                print(x[pareto1])
                print(y[pareto1])

                # s = 5 + 15 * (3 - np.clip(np.log10(losses2[mask]), -3.0, 0.0))
                # s = np.full(len(x), 30.0)
                # s[pareto2] *= 3
                alpha = np.full(len(x), 0.4)
                alpha[pareto2] = 1.0
                scatter = ax.scatter(x, y, color=color, marker=marker, label=geoinfo, edgecolor='none', alpha=alpha)
                plot = ax.plot(x[pareto1], y[pareto1], color=color)
                # ax.scatter(x[pareto2], y[pareto2], s=100, color=color, marker='o', edgecolor='none')

                if len(mark_ids):
                    ax.scatter(losses1[mask & is_marked],
                               utilizations[mask & is_marked],
                               color='none', marker='o', label=None, edgecolor='red',
                               s=200, zorder=100)

            ax.set_xscale('log')
            ax.set_ylabel("macro utilization", color='darkgreen')
            ax.set_ylim(0.0, 1.0)
            ax.spines['left'].set_edgecolor('darkgreen')
            ax.tick_params(axis='y', colors='darkgreen')
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
            ax.grid()

            # https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib
            legend = ax.legend()
            for handle in legend.legendHandles:
                # The legend symbol picks up the alpha from the samples
                # (probably the first one), which might be smaller than 1.0.
                # Here we fix that.
                handle.set_alpha(1.0)


        def _plot_ll(ax: Axes, losses1, losses2):
            nans = np.isnan(losses_macro1 + losses_macro2 + utilizations)
            for geoinfo, marker, color in \
                    zip(unique_geoinfos, MARKERS, get_default_rgba(len(unique_geoinfos))):
                mask = np.array([g == geoinfo for g in self.xgeoinfos])
                mask &= ~nans
                x = losses_macro1[mask]
                y = losses_macro2[mask]
                s = 10 + 100 * utilizations[mask]
                ax.scatter(x, y, s=s, color=color, marker=marker, label=geoinfo, edgecolor='none')

                pareto1 = pareto_front(-x, s)
                pareto2 = pareto_front(-y, s)
                ax.plot(x[pareto1], y[pareto1], color=color)
                ax.plot(x[pareto2], y[pareto2], color=color, linestyle='--')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(loss_label1)
            ax.set_xlabel(loss_label2)
            ax.grid()
            ax.legend()

        with Plot(path_fmt.format('avgmacro-1u2'), figsize=figsize) as (fig, ax):
            _plot_lu(ax, losses_macro1, losses_macro2)
            ax.set_xlabel(loss_label1)

        with Plot(path_fmt.format('avgmacro-2u1'), figsize=figsize) as (fig, ax):
            _plot_lu(ax, losses_macro2, losses_macro1)
            ax.set_xlabel(loss_label2)

        with Plot(path_fmt.format('avgmacro-12u'), figsize=figsize) as (fig, ax):
            _plot_ll(ax, losses_all1, losses_all2)

    def plot_utilization_and_loss_vs_t(
            self,
            *args,
            extra_multikey: Tuple[str, ...] = (),
            extra_label: str = "",
            stride: int = 50,
            macro_smoothing: float = 2000.0,
            **kwargs) -> Plot:
        from scipy.ndimage import gaussian_filter1d
        plot = super().plot_utilization_vs_t(
                *args, stride=stride, macro_smoothing=macro_smoothing, **kwargs)
        ax: Axes = plot.ax_utilization
        # ax3: Axes = plot.ax.twinx()

        colors = get_default_rgba(len(self.stats))
        for stat, color, label in zip(self.stats, colors, self.xticklabels):
            t = stat.postprocessed['runtime', 'metadata', 'timestep']
            stages = stat.postprocessed['runtime', 'metadata', 'stage']
            extra = stat.postprocessed[extra_multikey][:, 0]
            extra = np.where(stages == Stage.MACRO, extra, 0.0)
            # cycles = get_cycle_slices(stages)
            # extra_max = np.empty(len(cycles))
            # for i, (begin, end) in enumerate(cycles):
            #     extra_max[i] = extra[begin:end].mean()
            extra = gaussian_filter1d(extra, macro_smoothing)
            ax.plot(t[::stride], extra[::stride], label=label, linestyle='--', color=color)

        plot.ax_utilization.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.set_ylabel(extra_label)
        ax.set_ylim(1e-5, 1e-1)
        ax.set_yscale('log')
        # ax.spines['right'].set_position(('outward', 60))
        ax.legend()
        return plot


# TODO: Refactor, legacy.
def compute_diagnostics(dir: str, which: str):
    config: Config = adaled.load(os.path.join(dir, 'config.pt'))
    diag = adaled.AdaLEDDiagnostics.load(os.path.join(dir, 'diagnostics-000.pt'))
    data = diag.per_cycle_stats.data

    num_timesteps = 16000
    start_cycle = np.searchsorted(data['start_timestep'], config.led.client.max_steps - num_timesteps)
    data = data[start_cycle:]
    print(dir)
    macro_eff = data['macro_steps', :-1].sum() / (data['start_timestep', -1] - data['start_timestep', 0])
    print(f"Last {num_timesteps} timesteps, starting from timestep {data['start_timestep', 0]}, macro efficiency={macro_eff:.3f}")

    config_loss = config.autoencoder.loss
    ae_losses = data['losses', 'transformer_train', 'layers']

    out = {
        'macro_eff': macro_eff,
        f'{which}_weight': getattr(config_loss, f'{which}_weight'),
        'layers_mean': {key: {} for key in ae_losses.keys()},
        'layers_std': {key: {} for key in ae_losses.keys()},
    }
    for layer in ae_losses.keys():
        def _compute(key, factor):
            L: np.ndarray = ae_losses[layer, key] / factor
            mean = float(L.mean())
            std = float(L.std())
            out['layers_mean'][layer][key] = mean
            out['layers_std'][layer][key] = std
            print(f"    {layer}  {key:>10}: {mean:.6f} +- {std:.6f}  factor={factor}")

        _compute("v", 1.0)
        _compute(which, getattr(config_loss, f'{which}_weight'))
        _compute("pressure", config_loss.pressure_weight)

    return config, out


def plot_losses_wrt_vort(
        path: str,
        info: adaled.TensorCollection,
        which: str):
    with Plot(path, figsize=(6, 2.5)) as (fig, ax_v):
        ax_v: Axes
        ax_w: Axes = ax_v.twinx()
        ax_eff: Axes = ax_v.twinx()

        weight = info[f'{which}_weight']


        v_mean = info['layers_mean', 'layer0', 'v']
        v_std = info['layers_std', 'layer0', 'v']
        w_mean = info['layers_mean', 'layer0', which]
        w_std = info['layers_std', 'layer0', which]

        v_color, w_color = get_default_rgba(2)
        v_color = tuple(v_color)  # Suppress some random warnings.
        w_color = tuple(w_color)
        eff_color = 'green'

        x_threshold = weight[weight > 0].min()

        ax_v.fill_between(weight, v_mean - v_std, v_mean + v_std, facecolor=v_color, alpha=0.3)
        ax_w.fill_between(weight, w_mean - w_std, w_mean + w_std, facecolor=w_color, alpha=0.3)
        ax_v.plot(weight, v_mean, label="velocity", color=v_color)
        ax_w.plot(weight, w_mean, label=which, color=w_color)
        ax_eff.plot(weight, info['macro_eff'], label="macro efficiency", color=eff_color)
        ax_v.set_xlabel(f"{which} loss contribution weight")
        ax_v.set_xlim(0.0, weight.max())
        ax_v.set_xscale('symlog', linthresh=x_threshold)
        ax_v.set_ylabel("training velocity loss", color=v_color)
        ax_w.set_ylabel(f"training {which} loss", color=w_color)
        ax_v.set_ylim(0.0, 0.004)
        ax_w.set_ylim(0.0, 0.4 if which == 'vorticity' else 0.04)
        # ax_v.set_ylim(1e-4, 1e-0)
        # ax_v.set_yscale('log')

        # Mark a discontinuous x-scale. Extra bottom/top spines have to be hidden.
        # https://stackoverflow.com/questions/5656798/is-there-a-way-to-make-a-discontinuous-axis-in-matplotlib#66314835
        ax_v.scatter([0.5 * x_threshold] * 2, [0.0, 0.004], color='white', clip_on=False, zorder=100)
        ax_v.text(0.5 * x_threshold, 0.0,   "//", zorder=101, ha='center', va='center')
        ax_v.text(0.5 * x_threshold, 0.004, "//", zorder=101, ha='center', va='center')

        ax_v.spines['right'].set_edgecolor(eff_color)
        ax_v.spines['left'].set_edgecolor(v_color)
        ax_v.tick_params(axis='y', colors=v_color)

        ax_w.spines['bottom'].set_visible(False)
        ax_w.spines['right'].set_visible(False)
        ax_w.spines['top'].set_visible(False)
        ax_w.spines['left'].set_edgecolor(w_color)
        ax_w.spines['left'].set_position(('outward', 60))
        ax_w.tick_params(axis='y', colors=w_color)
        ax_w.yaxis.set_label_position('left')
        ax_w.yaxis.set_ticks_position('left')

        ax_eff.set_frame_on(False)
        ax_eff.set_ylabel("macro efficiency", color=eff_color)
        ax_eff.set_ylim(0.0, 0.8)
        ax_eff.tick_params(axis='y', colors=eff_color)
        ax_eff.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax_v.grid()


def create_parser():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--prefix', type=str, default='cmp-', help="output file path suffix")
    add('--suffix', type=str, default='', help="output file path suffix")
    # TODO: deprecate / refactor --which?
    add('--which', type=str, default='vorticity', help="type of loss to compare")
    # add('--what', type=str, choices=('pareto', 'violins'), nargs='+', required=True, help="what to plot")
    add('--xlabel-key', type=str, required=True,
        help="X-label. Options: dir, index, AE, vorticity_weight, "
             "derivatives_weight, sigma, cylinder_mask, config.<anything>, ")
    add('--mark', type=str, nargs='+', default=[], help="run dirs to emphasize")
    add('--kind', type=str, required=True, help="kind of plots to make",
        choices=('grouped-final', 'grouped-t', 'architecture-cmp', 'pareto', 'utilization'))
    add('dirs', type=str, nargs=argparse.REMAINDER, help="runs to compare")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = create_parser()
    args = parser.parse_args(argv)
    mark_ids = [args.dirs.index(mark) for mark in args.mark]

    plotter = CylComparisonPlotter(args.dirs, xlabel_key=args.xlabel_key)
    # plotter.plot_violin_force_errors(f'cmp-forces-{args.suffix}.png').finalize()
    # plotter.plot_violin(
    #         f'cmp-forces-transition-{args.suffix}.png',
    #         ('runtime', 'transition_qoi_errors', 'l2_first_warmup', 'cyl_forces'),
    #         "Post-transition force error").finalize()
    # plotter.plot_violin(
    #         f'cmp-forces-transition-check-{args.suffix}.png',
    #         ('runtime', 'transition_qoi_errors', 'l2_last_macro_only', 'cyl_forces'),
    #         "Pre-transition force error, validation for post-transition").finalize()
    # plotter.plot_violin(f'cmp-losses-layers-{args.suffix}.png',
    #                     ('postprocess', 'losses', 'micro_macro', 'layers'),
    #                     "autoencoder layers loss (all summed up)").finalize()
    # plotter.plot_violin(f'cmp-losses-rebuilt-rel-v-no-eps-{args.suffix}.png',
    #                     ('postprocess', 'errors', 'rel_v_no_eps', 'micro_macro'),
    #                     "autoencoder relative velocity loss (no eps)").finalize()

    # plotter.plot_violin(
    #         f'cmp-rvne-{args.suffix}.png',
    #         ('runtime', 'cmp_error', 'v'),
    #         "velocity field error $E$").finalize()
    # plotter.plot_execution_time(f'cmp-execution-time-no-overhead-{args.suffix}.png')
    # plotter.plot_utilization(f'cmp-utilization-{args.suffix}.png').finalize()

    # plotter.plot_utilization_and_loss_vs_t(
    #         f'cmp-utilization-t-rvne-{args.suffix}.png',
    #         F_title="Re",
    #         extra_multikey=('runtime', 'cmp_error', 'v'),
    #         extra_label="velocity relative MSE").finalize()

    if args.kind == 'grouped-t':
        main_grouped_t(args, plotter)
    elif args.kind == 'grouped-final':
        main_grouped_final(args, plotter)
    elif args.kind == 'architecture-cmp':
        import warnings
        warnings.warn("architecture-cmp is deprecated, use pareto")
        main_error_utilization_pareto(args, plotter, mark_ids)
    elif args.kind == 'pareto':
        main_error_utilization_pareto(args, plotter, mark_ids)
    elif args.kind == 'utilization':
        main_utilization(args, plotter)

    # plotter.plot_violin(
    #         f'cmp-vorticity-{args.suffix}.png',
    #         ('runtime', 'cmp_error', 'unweighted_loss', 'layers', 'layer1', 'vorticity'),
    #         "vorticity field error $E$").finalize()
    # plotter.plot_violin(
    #         f'cmp-F-{args.suffix}.png',
    #         ('runtime', '__qoi_error_l2'),
    #         "forces error",
    #         yclip=1e-4).finalize()

    # plotter.plot_utilization_vs_loss(
    #         f'cmp-utilization-vs-rvne-loss-{{}}{args.suffix}.png',
    #         ('postprocess', 'errors', 'rel_v_no_eps', 'micro_macro'),
    #         "mean relative MSE of velocity")



def main_grouped_t(args, plotter: CylComparisonPlotter):
    # seaborn-colorblind
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
              "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
    linestyles = ['-', ':', '--', '-.', '-', '-', '-', '-', '-', '-']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors, linestyle=linestyles)

    def _finalize(plot: Plot, is_Re: bool = False):
        if is_Re and plotter.stats[0].postprocessed['runtime', 'F'].max() > 1000:
            plot.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        plot.finalize()

    # Save tex version to pick up colors and line styles.
    # _finalize(plotter.plot_F_field_vs_t_grouped(
    #         f'{args.prefix}utilization-t-{args.suffix}.png:pdf:tex', F_title="Re"))
    if 'runtime' in plotter.stats[0].postprocessed:
        # _finalize(plotter.plot_F_field_vs_t_grouped(
        #         f'{args.prefix}utilization-t-cumulative-{args.suffix}.png:pdf', F_title="Re",
        #         multikey=('runtime', 'metadata', '__is_macro_cumulative'),
        #         ylim=(0.0, 0.6), ylabel="cumulative macro utilization"))
        # _finalize(plotter.plot_F_field_vs_t_grouped(
        #         f'{args.prefix}execution-time-t-cumulative-{args.suffix}.png:pdf', F_title="Re",
        #         multikey=('runtime', 'metadata', '__execution_time_cumulative_no_overhead'),
        #         ylabel="execution time [s]"))
        # _finalize(plotter.plot_F_field_vs_t_grouped(
        #         f'{args.prefix}rvne-t-{args.suffix}.png:pdf', F_title="Re",
        #         multikey=('runtime', 'cmp_error', '__v_when_macro'),
        #         ylabel="relative velocity MSE", ylim=(1e-3, 1e-1), yscale='log',
        #         # yplot_kwargs={'linestyle': '--'},
        #         smoothing_before=0, smoothing_after=1000.0))

        _finalize(plotter.plot_F_vs_t_grouped(
                f'{args.prefix}Re-t-{args.suffix}.png:pdf', F_title="Re"), is_Re=True)
        _finalize(plotter.plot_field_vs_t_grouped(
                f'{args.prefix}rvne-t-{args.suffix}.png:pdf',
                multikey=('runtime', 'cmp_error', '__v_when_macro'),
                ylabel="rel. velocity MSE", ylim=(1e-3, 1e-1), yscale='log',
                # yplot_kwargs={'linestyle': '--'},
                smoothing_before=0, smoothing_after=1000.0))
        _finalize(plotter.plot_field_vs_t_grouped(
                f'{args.prefix}EF-t-{args.suffix}.png:pdf',
                multikey=('runtime', '__qoi_error_norm_Fcyl_when_macro',),
                ylabel="$E_F$", ylim=(1e-3, 1e-0), yscale='log',
                # yplot_kwargs={'linestyle': '--'},
                smoothing_before=0, smoothing_after=1000.0))
        _finalize(plotter.plot_field_vs_t_grouped(
                f'{args.prefix}utilization-t-{args.suffix}.png:pdf',
                multikey=('runtime', 'metadata', '__is_macro')))


def main_grouped_final(args, plotter: CylComparisonPlotter):
    plotter.plot_final_grouped(
            f'{args.prefix}utilization-{args.suffix}.png:pdf',
            lambda stat: stat.macro_util[-1],
            ylabel="macro utilization",
            ylim=(0.0, 1.0)).finalize()
    post = plotter.stats[0].postprocessed
    if 'runtime' in post:
        plotter.plot_final_grouped(
                f'{args.prefix}execution-time-{args.suffix}.png:pdf',
                lambda stat: stat.postprocessed[
                    'runtime', 'metadata', '__execution_time_cumulative_no_overhead', -1],
                ylabel="total execution time [s]", sem=True).finalize()
    if 'cmp_error' in post.get('runtime', ()):
        plotter.plot_final_grouped(
                f'{args.prefix}rvne-{args.suffix}.png:pdf',
                lambda stat: stat.postprocessed['runtime', 'cmp_error', '__v_when_macro', -1],
                ylabel="relative velocity MSE", ylim=(1e-3, 1e-1),
                yscale='log').finalize()


def main_error_utilization_pareto(args, plotter: CylComparisonPlotter, mark_ids):
    plotter.plot_utilization_vs_loss_double_pareto(
            f'cmp-utilization-vs-rvne-vs-forces-{{}}{args.suffix}.png:pdf',
            ('runtime', 'cmp_error', 'v'),
            ('runtime', '__qoi_error_norm_Fcyl'),
            "average velocity relative MSE $E$",
            "average normalized force error $E_F$",
            mark_ids=mark_ids)


def main_utilization(args, plotter: CylComparisonPlotter):
    plotter.plot_utilization(f'cmp-utilization-{args.suffix}.png').finalize()
    plotter.plot_execution_time(f'cmp-execution-time-{args.suffix}.png')

if __name__ == '__main__':
    main()
