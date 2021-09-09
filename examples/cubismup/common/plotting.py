from adaled import TensorCollection
from adaled.led import AdaLEDStage as Stage
from adaled.plotting.base import Task, Plotter
from adaled.plotting.plot_dataset import DatasetPlotter
from adaled.plotting.plot_diagnostics import LOSSES_DEFAULT, DiagnosticsPlotter
from adaled.plotting.plot_record import \
        FIGWIDTH, MergedRecordsPlotter, StageInfo, add_stage_shaded_regions
from adaled.plotting.plot_transformer import LayerTracer, TransformerPlotter
from adaled.plotting.plots import Axes, Figure, Plot, mpl, plt
from adaled.plotting.plots_multiresolution import plot_multiresolution_1d_contributions
from adaled.plotting.utils import darken_color, divider_colorbar, fade_color, get_default_rgba
from adaled.postprocessing.record import get_cycle_slices
from .config import CombinedConfigBase, MicroConfigBase
from .micro import AutoencoderReconstructionLayer, MicroStateHelper
from .loss import LayerLoss, VorticityLayer
from .movie import Movie2DPlotter, Movie3DPlotter, plot_movie_2d
from .postprocess import parse_execution_time_from_stderr
from .utils_2d import compute_divergence_2d, compute_vorticity_2d, \
        stream_function_to_velocity
from .utils_3d import divergence_3d, curl
from .setup import CombinedConfigBase
import adaled

import numpy as np
import torch

from typing import List, Optional, Sequence, Tuple
import argparse
import glob
import re
import warnings

class PlotMultiresolution1D(Plot):
    """Plot multiresolution training and reconstruction bounds as a 1D slice."""
    def __init__(
            self,
            path: str,
            config: CombinedConfigBase,
            d: int,
            *args,
            figsize=(12, 4),
            **kwargs):
        super().__init__(path, *args, figsize=figsize, **kwargs)
        ax: Axes = self.ax

        mask = [False] * len(config.micro.cells)
        mask[d] = True
        self.mr = mr = config.micro.make_multiresolution(dim_mask=mask)

        colors = get_default_rgba(len(mr.layers))
        plot_multiresolution_1d_contributions(ax, path, mr, "xyz"[d], colors)

        # This is a submasked MR, we don't include the potential obstacle mask, only the MR mask.
        loss_weights = config.compute_multiresolution_submask_weights(mr)
        loss_weights[1] = 1 - loss_weights[1]

        for i, (color, layer) in enumerate(zip(colors, mr.layers)):
            xp = np.arange(*layer.begin, *layer.end, *layer.stride)
            ax.plot(xp, loss_weights[i], color=color, linestyle='--', label=f"layer #{i} loss")

    def __enter__(self) -> Tuple['PlotMultiresolution1D', Figure, Axes]:
        return self, self.fig, self.ax

    def plot_hatched_obstacle(self, lo: float, hi: float):
        # with plt.rc_context({'hatch.linewidth': 5}):
        plt.rc('hatch', linewidth=4)
        # https://stackoverflow.com/questions/5195466/matplotlib-does-not-display-hatching-when-rendering-to-pdf
        self.ax.axvspan(lo, hi, facecolor='black', edgecolor='none', alpha=0.15)
        self.ax.axvspan(lo, hi, facecolor='none', edgecolor='black', alpha=0.15, hatch='//')


def plot_multiresolution_weights_2d(
        path: str, config: CombinedConfigBase, layer: int):
    """Plot the weight matrix for one layer. Useful for visualizing cyl_weight
    in cyl2d."""
    mr = config.micro.make_multiresolution()
    weight = config.compute_multiresolution_weights(mr)[layer]
    if weight is None:
        weight = np.full(mr.layers[layer].downscaled_shape, 1.0, dtype=np.float32)

    plot = Plot(path)
    ax = plot.ax
    im = ax.imshow(weight, interpolation='nearest')
    divider_colorbar(plot.fig, im, ax)
    ax.set_title(f"layer #{layer} weight")
    return plot


def plot_postprocessed_errors(path: str, data: TensorCollection):
    """Plot micro-macro velocity error and the error between two consecutive
    micro frames as the baseline."""
    plot = Plot(path)
    ax = plot.ax

    colors = get_default_rgba(4)
    is_macro_only = data['metadata', 'stage'] == Stage.MACRO

    t = data['metadata', 'timestep']
    errors = data['postprocess', 'errors', 'rel_v_no_eps']
    ax.plot(t, errors['consecutive'], color=colors[0], label="baseline")
    ax.plot(t, np.ma.array(errors['micro_macro'], mask=~is_macro_only),
            color=colors[1], label="surrogate error (macro-only)")
    ax.plot(t, np.ma.array(errors['micro_macro'], mask=is_macro_only),
            color=fade_color(colors[1], 0.8), label="surrogate error (rest)")
    if 'full_micro_scaling' in errors:
        ax.plot(t, errors['full_micro_scaling'], color='black',
                linestyle='-', label="micro scaling error")
        ax.plot(t, errors['full_micro_consecutive'], color=colors[2],
                linestyle='--', label="baseline (full resolution)")
        ax.plot(t, errors['full_micro_downscaled_macro'], color=colors[3],
                linestyle='--', label="surrogate error (full resolution)")
    ax.set_xlim(t[0], round(t[-1], -2))
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    return plot


def plot_postprocessed_losses(path: str, data: TensorCollection, with_ref: bool = False):
    """Plot micro-macro validation loss and the loss between two consecutive
    micro frames as the baseline."""
    plot = Plot(path)
    ax = plot.ax

    colors = get_default_rgba(4)
    is_macro_only = data['metadata', 'stage'] == Stage.MACRO

    t = data['metadata', 'timestep']
    losses = data['postprocess', 'losses']
    total = {key: sum(parts.allvalues()) for key, parts in losses.items()}
    if with_ref:
        ax.plot(t, total['ref_frame'], color='gray', linewidth=1.0,
                label="loss wrt reference frame")

    ax.plot(t, total['consecutive'], color=colors[0], label="baseline")
    ax.plot(t, np.ma.array(total['micro_macro'], mask=~is_macro_only),
            color=colors[1], label="surrogate loss (macro-only)")
    ax.plot(t, np.ma.array(total['micro_macro'], mask=is_macro_only),
            color=fade_color(colors[1], 0.8), label="surrogate loss (rest)")
    ax.set_xlim(t[0], round(t[-1], -2))
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    return plot


def plot_postprocessed_runtime_qoi(path: str, data: TensorCollection):
    """Plot `micro_state_qoi` and `macro_qoi` from results computed by
    `./postprocess.sh.
    """
    stages = data['metadata', 'stage']
    is_accepted = stages == Stage.MACRO

    def normalize_to_2d(multikey, v: np.ndarray):
        assert 2 <= v.ndim <= 3 and v.shape[1] == 1, (multikey, v.shape)
        return v if v.ndim == 2 else v[:, 0, :]

    micro_qoi = data['micro_state_qoi']
    macro_qoi = data['macro_qoi']
    micro_qoi: TensorCollection = micro_qoi.named_map(normalize_to_2d)
    macro_qoi: TensorCollection = macro_qoi.named_map(normalize_to_2d)
    macro_qoi = macro_qoi.clone()
    macro_qoi[~is_accepted] = np.nan

    t = data['metadata', 'timestep']

    plot = Plot(path, figsize=(FIGWIDTH, 1.35))
    ax = plot.ax

    cycles = get_cycle_slices(stages)
    for begin, end in cycles[::2]:  # Darken every other cycle.
        ax.axvspan(t[begin], t[min(end, len(t) - 1)],
                   facecolor='black', alpha=0.06)
    # add_stage_shaded_regions(
    #         ax, t, stages, alpha=0.10,
    #         stage_infos={Stage.MACRO: StageInfo('green', "macro")})

    total_lines = sum(micro_qoi.map(lambda x: x.shape[1]).allvalues())
    colors = get_default_rgba(total_lines)
    c = 0
    for multikey, value1 in micro_qoi.allitems():
        value2 = macro_qoi[multikey]
        key = "/".join(multikey)
        for j in range(value1.shape[1]):
            ax.plot(t, value1[:, j], linewidth=1.5, color=colors[c],
                    label=f"micro {key}/{j}")
            ax.plot(t, value2[:, j], linewidth=1.5,
                    color='black', linestyle='--',
                    label=f"macro {key}/{j}")
            c += 1
    ax.set_xlim(t.min(), round(t.max(), -1))
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$F_{\mathrm{cyl}}$")
    # ax.grid()
    # ax.legend()

    return plot


def _extend_ref(ref: np.ndarray, ref_extension: int, length: int):
    new = np.empty(length, dtype=ref.dtype)
    new[:len(ref)] = ref
    if ref_extension > 0:
        for offset in range(len(ref), length, ref_extension):
            tmp = new[offset : offset + ref_extension]
            tmp[:] = ref[-ref_extension:][:len(tmp)]
    else:
        new[len(ref):] = np.nan
    return new


def plot_report_execution_time(
        path: str,
        data: TensorCollection,  # metadata
        smooth_sigma: float = 500.0,
        ref: Optional[np.ndarray] = None,
        ref_extension: int = 0,
        exec_time: Optional[np.ndarray] = None,
        speedup_color: str = 'green',
        details: bool = True,
        stride: int = 200):
    from adaled.utils.arrays import masked_gaussian_filter1d
    timestep = data['timestep']

    if ref is not None:
        if len(ref) < len(timestep):
            ref = _extend_ref(ref, ref_extension, len(timestep))
        ref[0] = np.nan
        ref = masked_gaussian_filter1d(ref, smooth_sigma, set_nan=True)

    if exec_time is not None:
        assert exec_time.ndim == 2 and exec_time.shape[1] == 2, exec_time.shape
        exec_time_cup = masked_gaussian_filter1d(exec_time[:, 0], smooth_sigma, set_nan=True)
        overhead = masked_gaussian_filter1d(
                exec_time[:, 1] - exec_time[:, 0], smooth_sigma, set_nan=True)

    with Plot(path, figsize=(FIGWIDTH, 1.5)) as (fig, ax1):
        ax1: Axes

        time = data['execution_time']

        is_macro = data['stage'] == Stage.MACRO
        time_avg = masked_gaussian_filter1d(time, 2 * smooth_sigma)

        def _print(y, label: str, suffix: str = ""):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                mean = np.nanmean(y)
                std = np.nanstd(y)
                cnt = (~np.isnan(y)).sum()
                print(f"{label:>20}: mean={mean:5f} std={std:5f}  N={cnt}{suffix}")

        def _plot(x, y, *args, label: str, **kwargs):
            ax1.plot(x, y, *args, label=label, **kwargs)
            _print(y, label)
            if ref_extension > 0:
                _print(y[-ref_extension:], label, suffix=f" (last {ref_extension} steps)")

        _plot(timestep, time_avg, label="average", linewidth=0.7, color='black')
        if details:
            time_rest = time.copy()
            time_macro = time.copy()
            time_rest[is_macro] = np.nan
            time_macro[~is_macro] = np.nan
            masked_gaussian_filter1d(time_rest, smooth_sigma, out=time_rest, set_nan=True)
            masked_gaussian_filter1d(time_macro, smooth_sigma, out=time_macro, set_nan=True)
            _plot(timestep, time_rest, label="rest", linewidth=1.0, color='black', alpha=0.5)
            _plot(timestep, time_macro, label="macro-only", linewidth=1.0, color='black', alpha=0.5, zorder=10, clip_on=False)
        if exec_time is not None:
            x = np.arange(len(exec_time_cup))
            _plot(x, exec_time_cup, label="CUP2D only", linestyle='--', color='darkgray')
            _plot(x, overhead, label="overhead", linestyle=':', color='darkgray')

        ax1.grid(color='lightgray')
        ax1.set_xlabel("time step")
        ax1.set_xlim(0, round(timestep.max(), -2))
        ax1.set_ylabel("execution time [s]")
        ax1.set_ylim(bottom=0.0, top=1.5)
        ax1.set_axisbelow(True)

        if ref is not None:
            ax1.fill_between(timestep[1::stride], 0.0, time_avg[1::stride],
                             color='orange', alpha=0.18)
            ax1.fill_between(timestep[1::stride], time_avg[1::stride],
                             ref[:len(timestep) - 1:stride], color='green', alpha=0.18)
            _plot(timestep[1::stride], ref[:len(timestep) - 1:stride],
                  label="no AdaLED", linewidth=0.7, color='black')
            # ax1.text(timestep[-1] * 0.050, ref[1:].max(), "without AdaLED",
            #          ha='left', va='bottom')
            # ax1.text(timestep[-1] * 0.999, time_avg[-len(time_avg) // 10:].max(), "with AdaLED",
            #          ha='right', va='bottom')

        if ref is not None and False:
            ax2: Axes = ax1.twinx()

            speedup = ref[:len(timestep)] / time_avg
            ax2.plot(timestep, speedup, label="speedup", color=speedup_color, zorder=15)

            ax1.spines['right'].set_visible(False)
            ax2.set_ylabel("total speedup", color=speedup_color)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_edgecolor(color=speedup_color)
            ax2.spines['top'].set_visible(False)
            ax2.tick_params(axis='y', colors=speedup_color)
            ax2.set_ylim(bottom=0.0, top=5.0)


def plot_report_validation_error(
        path: str,
        metadata: TensorCollection,
        error: np.ndarray,
        threshold: Optional[float] = None,
        ylabel: str = "error $E$",
        smooth_sigma: float = 500.0,
        ylim: Tuple[Optional[float], Optional[float]] = (None, None)):
    """Plot given per-timestep error.

    The errors are expected to go from small to large values on each cycle.
    For sufficiently large records, the lines themselves will merge.
    Some PDF rendering software are fine with so many line segments, some freeze.

    Thus, for large records, plots are compressed and plotted using
    ax.fill_between instead of ax.plot.
    """
    from adaled.utils.arrays import masked_gaussian_filter1d
    timestep = metadata['timestep']
    compress = len(timestep) >= 50000
    if compress:
        print(f"Using compressed plot for {ylabel}.")

    plot = Plot(path, figsize=(FIGWIDTH, 1.5))
    ax: Axes = plot.ax

    is_macro = metadata['stage'] == Stage.MACRO
    error = np.where(is_macro, error, np.nan)

    def _plot(x, y, *, color, **kwargs):
        if compress:
            K = 200
            x = x[::K]
            if len(y) % K:
                y = np.pad(y, (0, K - len(y) % K), constant_values=np.nan)
            y = y.reshape(-1, K)[:len(x), :]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                ymin = np.nanmin(y, axis=-1)
                ymax = np.nanmax(y, axis=-1)
            # We need to manually create "stairs", because alpha facecolor and
            # edgecolor do not look good together.
            ymin = np.repeat(ymin, 2)[:-1]
            ymax = np.repeat(ymax, 2)[:-1]
            x = np.repeat(x, 2)[1:]
            ax.fill_between(x, ymin, ymax, facecolor=color, edgecolor=None, **kwargs)
        else:
            ax.plot(x, y, color=color, **kwargs)

    _plot(timestep, error, color=(1.0, 0.7, 0.7, 0.85),
          label="validation error (macro-only)")
    if threshold is not None:
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1.0,
                   label="threshold")

    error = masked_gaussian_filter1d(error, smooth_sigma)
    ax.plot(timestep[::50], error[::50], color=darken_color('red', 0.5),
            label="smoothed error (macro-only)")

    ax.grid(color='lightgray')
    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(round(timestep.min(), -2), round(timestep.max(), -2))
    if any(ylim):
        ax.set_ylim(*ylim)
    ax.set_yscale('log')
    ax.set_axisbelow(True)

    print(f"Average macro-only {ylabel}: {np.nanmean(error)} +- {np.nanstd(error)}")

    return plot


def plot_report_validation_error_detail(
        path: str,
        config: CombinedConfigBase,
        metadata: TensorCollection,
        error: np.ndarray,
        uncertainty: np.ndarray):
    from adaled.utils.arrays import masked_gaussian_filter1d
    timestep = metadata['timestep']

    plot = Plot(path, figsize=(FIGWIDTH, 1.5))
    ax: Axes = plot.ax


    is_macro = metadata['stage'] == Stage.MACRO
    cycles = get_cycle_slices(metadata['stage'])
    for begin, end in cycles[::2]:  # Darken every other cycle.
        ax.axvspan(timestep[begin], timestep[min(end, len(timestep) - 1)],
                   facecolor='black', alpha=0.06)

    ax.axhline(config.led.criteria.max_cmp_error,
               color='red', linestyle='--', linewidth=1.0,
               label="threshold $E_{\mathrm{max}}$")
    ax.axhline(config.led.criteria.max_uncertainty,
               color='orange', linestyle='--', linewidth=1.0,
               label="threshold $\sigma_{\mathrm{max}}$")
    ax.plot(timestep, np.where(is_macro, np.nan, error),
            color='red', linestyle=':', label="validation error (macro-only)")
    ax.plot(timestep, np.where(is_macro, error, np.nan),
            color='red', label="validation error (macro-only)")
    ax.plot(timestep, np.where(is_macro, np.nan, uncertainty),
            color=fade_color('orange', 0.9), label="uncertainty $\sigma$ (macro-only)")
    ax.plot(timestep, np.where(is_macro, uncertainty, np.nan),
            color='orange', label="uncertainty $\sigma$ (macro-only)")

    # ax.grid(color='lightgray')
    ax.set_xlabel("time step")
    ax.set_xlim(round(timestep.min(), -2), round(timestep.max(), -2))
    ax.set_ylabel("$\sigma^2$ and $E$")
    ax.set_yscale('log')
    ax.set_axisbelow(True)

    return plot


def _extract_fields(
        config: CombinedConfigBase,
        layer: int,
        channels1: np.ndarray,
        channels2: np.ndarray,
        shape_yx: Tuple[int, int]) -> Tuple:
    channels1 = channels1[np.newaxis]
    channels2 = channels2[np.newaxis]
    helper0 = MicroStateHelper(config.micro)
    helper1 = AutoencoderReconstructionLayer(config.micro)
    v0 = helper0.layer_to_velocity(channels1)[0]
    v1 = helper1.layer_to_velocity(layer, channels2)[0]
    vort0 = helper0.layer_to_vorticity_with_boundary(layer, channels1)[0]
    vort1 = helper1.v_to_vorticity_with_boundary(layer, v1[np.newaxis])[0]

    to_divergence = compute_divergence_2d if helper0.ndim == 2 else divergence_3d
    div0 = to_divergence(v0, helper0.hs[layer])
    div1 = to_divergence(v1, helper1.hs[layer])
    p0 = helper0.layer_to_pressure(channels1)
    p1 = helper1.layer_to_pressure(channels2)
    stream1 = helper1.layer_to_stream(channels2)
    if p0 is not None:
        p0 = p0[0]
    if p1 is not None:
        p1 = p1[0]
    if stream1 is not None:
        stream1 = stream1[0]

    if helper0.ndim == 3:
        nz = v0.shape[1]
        v0 = v0[:2, nz // 2, :, :]
        v1 = v1[:2, nz // 2, :, :]
        vort0 = vort0[2, nz // 2, :, :]
        vort1 = vort1[2, nz // 2, :, :]
        div0 = div0[nz // 2, :, :]
        div1 = div1[nz // 2, :, :]
        if p0 is not None:
            p0 = p0[nz // 2, :, :]
        if p1 is not None:
            p1 = p1[nz // 2, :, :]
        if stream1 is not None:
            stream1 = stream1[2, nz // 2 + 1, :, :]

    assert v0.shape == (2,) + shape_yx, (v0.shape, config.micro.cells, shape_yx)
    assert v1.shape == (2,) + shape_yx, (v1.shape, config.micro.cells, shape_yx)
    assert div0.shape == v0[0].shape, (div0.shape, v0.shape)
    assert div1.shape == v1[0].shape, (div1.shape, v1.shape)
    assert vort0.shape == v0[0].shape, (vort0.shape, v0.shape)
    assert vort1.shape == v1[0].shape, (vort1.shape, v1.shape)
    if p0 is not None:
        assert p0.shape == v0[0].shape, (p0.shape, v0.shape)
    if p1 is not None:
        assert p1.shape == v1[0].shape, (p1.shape, v1.shape)
    if stream1 is not None:
        assert stream1.shape == (v1[0].shape[0] + 2, v1[0].shape[1] + 2), \
               (stream1.shape, v1.shape)

    return (
        np.stack([v0, v1]),
        np.stack([vort0, vort1]),
        np.stack([div0, div1]),
        np.stack([p0, p1]) if p0 is not None and p1 is not None else (p0, p1),
        stream1,
    )


def plot_reconstruction(
        path: str,
        config: CombinedConfigBase,
        layer: int,
        state: adaled.TensorCollection,
        state_idx: int,
        transformer: adaled.Transformer):
    x_micro = state['x', 'layers', f'layer{layer}']
    with torch.no_grad():
        x_macro = transformer.to_latent_and_back(state[None, ...])[0].numpy()
    x_macro = x_macro['layers', f'layer{layer}']

    ndim = len(config.micro.cells)
    mr = config.micro.make_multiresolution()
    loss = LayerLoss(config, mr)
    layer_h = mr.layers[layer].stride[0] * config.micro.compute_h()

    shape_yx = mr.layers[layer].downscaled_shape[-2:]
    vs, vorts, divs, ps, stream_macro = _extract_fields(config, layer, x_micro, x_macro, shape_yx)

    max_v = config.compute_max_v()
    L = config.micro.extent
    v_scale = 0.7 * max_v
    div_scale = 1e0 * max_v / L
    vort_scale = 1e2 * max_v / L
    p_scale = 0.3 * max_v ** 2
    if ndim == 2:
        stream_scale = 0.015 * max_v * L
    else:
        stream_scale = 0.015 * max_v * L
    norm_v = mpl.colors.Normalize(vmin=-v_scale, vmax=+v_scale)
    norm_v_error = mpl.colors.Normalize(vmin=0.0, vmax=+0.1 * v_scale)
    norm_div = mpl.colors.Normalize(vmin=-div_scale, vmax=+div_scale)
    norm_vort = mpl.colors.Normalize(vmin=-vort_scale, vmax=+vort_scale)
    norm_vort_error = mpl.colors.Normalize(vmin=0.0, vmax=+0.1 * vort_scale)
    norm_p = mpl.colors.Normalize(vmin=-p_scale, vmax=+p_scale)
    norm_p_error = mpl.colors.Normalize(vmin=0.0, vmax=+0.1 * p_scale)
    norm_stream = mpl.colors.Normalize(vmin=-stream_scale, vmax=+stream_scale)

    losses = loss.layer_loss(layer, x_macro[np.newaxis], x_micro[np.newaxis])
    losses = losses.numpy()[0]
    if config.micro.predict_pressure:
        p_msg = f"p: {ps.mean():.4g}+-{ps.std():.4g}"
    else:
        p_msg = ""
    print(f"layer={layer} v: {vs.min(axis=(0, 2, 3))}..{vs.max(axis=(0, 2, 3))}  "
          f"vort: +-{vorts.std():.4g}  "
          f"{p_msg}  "
          f"max_v={max_v:.4f}  L={L}  max_v/L={max_v/L:.4f}  "
          f"losses={losses:.3g}")

    cmp_kwargs = {
        'cmap': 'RdBu',
        'interpolation': 'antialiased',
        'interpolation_stage': 'rgba',
    }

    plot = Plot(path, nrows=4 + bool(config.micro.predict_pressure), ncols=3,
                figsize=(16.0, 10.0), dpi=150)
    fig = plot.fig
    axs = plot.ax

    for i, v, vort, div, p in zip(range(2), vs, vorts, divs, ps):
        im0 = axs[0, i].imshow(v[0, :, :], norm=norm_v, cmap='RdBu')
        im1 = axs[1, i].imshow(v[1, :, :], norm=norm_v, cmap='RdBu')
        im2 = axs[2, i].imshow(vort, norm=norm_vort, **cmp_kwargs)
        imd = axs[-1, i].imshow(div, norm=norm_div, **cmp_kwargs)
        divider_colorbar(fig, im0, axs[0, i])
        divider_colorbar(fig, im1, axs[1, i])
        divider_colorbar(fig, im2, axs[2, i])
        divider_colorbar(fig, imd, axs[-1, i])
        if p is not None:
            im3 = axs[3, i].imshow(p, norm=norm_p, **cmp_kwargs)
            divider_colorbar(fig, im3, axs[3, i])

    # Plot error.
    v_error = np.abs(vs[1] - vs[0][:2, :, :])
    vort_error = np.abs(vorts[1] - vorts[0])
    im0 = axs[0, 2].imshow(v_error[0, :, :], norm=norm_v_error, cmap='Reds')
    im1 = axs[1, 2].imshow(v_error[1, :, :], norm=norm_v_error, cmap='Reds')
    im2 = axs[2, 2].imshow(vort_error, norm=norm_vort_error, cmap='Reds')
    divider_colorbar(fig, im0, axs[0, 2])
    divider_colorbar(fig, im1, axs[1, 2])
    divider_colorbar(fig, im2, axs[2, 2])
    if config.micro.predict_pressure:
        p_error = np.abs(ps[1] - ps[0])
        im3 = axs[3, 2].imshow(p_error, norm=norm_p_error, cmap='Reds')
        divider_colorbar(fig, im3, axs[3, 2])

    vorticity = "vorticity" if ndim == 2 else "$\omega_z$"
    axs[0, 0].set_title("$u_x$ (dataset)")
    axs[1, 0].set_title("$u_y$ (dataset)")
    axs[2, 0].set_title(f"{vorticity} (dataset)")
    axs[-1, 0].set_title("divergence (dataset)")
    axs[0, 1].set_title("$u_x$ (reconstructed)")
    axs[1, 1].set_title("$u_y$ (reconstructed)")
    axs[2, 1].set_title(f"{vorticity} (reconstructed)")
    axs[-1, 1].set_title("divergence (reconstructed)")
    axs[0, 2].set_title("$u_x$ (reconstruction error)")
    axs[1, 2].set_title("$u_y$ (reconstruction error)")
    axs[2, 2].set_title(f"{vorticity} (reconstruction error)")
    if config.micro.predict_pressure:
        axs[3, 0].set_title("pressure (dataset)")
        axs[3, 1].set_title("pressure (reconstruction)")
        axs[3, 2].set_title("pressure (reconstruction error)")

    if stream_macro is not None:
        mean = stream_macro.mean()
        stream_macro -= mean
        im = axs[-1, 2].imshow(stream_macro, norm=norm_stream, cmap='RdBu')
        divider, cax, cbar = divider_colorbar(fig, im, axs[-1, 2])
        cax.axhline(mean)
        if ndim == 2:
            axs[-1, 2].set_title("stream function")
        else:
            axs[-1, 2].set_title("potential function (z)")
    else:
        fig.delaxes(axs[-1, 2])

    for ax in axs[:, 1:].ravel():
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    for ax in axs[:-1, :].ravel():
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])

    fig.suptitle(f"reconstructed fields for dataset "
                 f"state #{state_idx}, layer #{layer}\n"
                 f"(vorticity and divergence scale is "
                 f"max_v/L={max_v:.3g}/{L:.3g}={max_v/L:.3g})",
                 horizontalalignment='center')

    return plot


class CUPPlotterBase(Plotter):
    PLOT_MERGED_LAYERS: bool

    def load_postprocessed_or_skip(self) -> List[Tuple[int, TensorCollection]]:
        """Return a list of (rank, data)."""
        if not hasattr(self, 'postprocessed'):
            paths = glob.glob('postprocessed-*.pt')
            pattern = re.compile(r'postprocessed-(\d+).pt')
            self.postprocessed = []
            for path in paths:
                match = pattern.match(path)
                if match:
                    rank = int(match.group(1))
                    self.postprocessed.append((rank, adaled.load(path)))
        if not self.postprocessed:
            self.skip("postprocessed results not found, run postprocess.sh")
        return self.postprocessed

    def load_postprocessed_runtime_or_skip(self) -> List[Tuple[int, TensorCollection]]:
        """Return a list of (rank, data)."""
        if not hasattr(self, 'postprocessed_runtime'):
            paths = glob.glob('postprocessed-runtime-*.pt')
            pattern = re.compile(r'postprocessed-runtime-(\d+).pt')
            self.postprocessed_runtime = out = []
            for path in paths:
                match = pattern.match(path)
                if match:
                    rank = int(match.group(1))
                    out.append((rank, adaled.load(path)))
        if not self.postprocessed_runtime:
            self.skip("postprocessed-runtime results not found, run ./postprocess.sh")
        return self.postprocessed_runtime

    def movie_func(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument
        add('--movie-frames', nargs='*', type=int, default=[-1],
            help="list of movie frames to plot as images")
        # TODO: --dpi could be specified on a global level somehow
        add('--dpi', type=int, default=100, help="DPI for the movie images")

    def tasks_last_movie_frame(self, report: bool = False, prefix: str = '', ext: str = 'png'):
        report = True
        ext = 'pdf'
        from .movie import tasks_all_rank_movies_2d
        config: Config = self.context.load_config()
        suffix_fmt = '{rank:03d}-{{frame:07d}}.' + ext
        kwargs = {
            'report': report,
            'dpi': self.args.dpi,
            'frames': self.args.movie_frames,
            'movie_func': self.movie_func,
        }
        if self.PLOT_MERGED_LAYERS:
            yield from tasks_all_rank_movies_2d(
                    f'{prefix}movie-{suffix_fmt}', config, **kwargs)
        for layer in range(len(config.micro.multiresolution)):
            yield from tasks_all_rank_movies_2d(
                    f'{prefix}movie-layer{layer}-{suffix_fmt}', config,
                    layer=layer, **kwargs)

    def tasks_postprocessed_errors(self):
        for rank, data in self.load_postprocessed_or_skip():
            yield Task(plot_postprocessed_errors, f'postprocess-error-{rank:03d}.png', data)

    def tasks_postprocessed_losses(self):
        for rank, data in self.load_postprocessed_or_skip():
            yield Task(plot_postprocessed_losses, f'postprocess-loss-{rank:03d}.png', data)
        if 'ref_frame' in data['postprocess', 'losses']:
            for rank, data in self.load_postprocessed_or_skip():
                yield Task(plot_postprocessed_losses,
                           f'postprocess-loss-with-ref-{rank:03d}.png', data, with_ref=True)
        else:
            print("ref_frame not found, rerun postprocessing with --ref-frame to "
                  "compute x_micro loss with respect to a specific frame.")

    def tasks_postprocessed_runtime_qoi(self):
        s = self.args.record_slice
        for rank, data in self.load_postprocessed_runtime_or_skip():
            if 'macro_qoi' in data:
                if s:
                    data = data[s[0]:s[1]:s[2]]
                yield Task(plot_postprocessed_runtime_qoi,
                           f'postprocess-runtime-qoi-{rank:03d}.png:pdf', data)

    def tasks_report_last_movie_frame(self):
        yield from self.tasks_last_movie_frame(report=True, prefix='report-', ext='pdf')


class CUPPlotter2D(CUPPlotterBase):
    PLOT_MERGED_LAYERS: bool = True

    def movie_func(self, *args, **kwargs):
        # FIXME: constructing movie plotter like this is hacky.
        return Movie2DPlotter(self.args).movie_func(*args, **kwargs)

    def tasks_multiresolution_weights(self):
        config: CombinedConfigBase = self.context.load_config()
        for layer in range(len(config.micro.multiresolution)):
            yield Task(plot_multiresolution_weights_2d,
                       f'multiresolution-weight-{layer}.png', config, layer)


class CUPPlotter3D(CUPPlotterBase):
    PLOT_MERGED_LAYERS: bool = False  # Not supported.

    def movie_func(self, *args, **kwargs):
        # FIXME: constructing movie plotter like this is hacky.
        return Movie3DPlotter(self.args).movie_func(*args, **kwargs)


class DatasetPlotterEx(DatasetPlotter):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument
        add('--dataset-seed', type=int, default=12345)
        add('--num-dataset-states', type=int, default=5)

    def set_up(self):
        super().set_up()

    def tasks_reconstruction(self):
        states = self.dataset.as_states()
        config: CombinedConfigBase = self.context.load_config()
        transformer = adaled.load('transformer-latest.pt')
        rng = np.random.default_rng(self.args.dataset_seed)
        if not len(states):
            return
        selection = rng.choice(len(states), self.args.num_dataset_states)
        for i, idx in enumerate(selection):
            for layer in range(len(config.micro.multiresolution)):
                yield Task(plot_reconstruction,
                           f'state-div-{layer}-{i:03d}.png:pdf', config, layer,
                           states['trajectory', idx], idx, transformer)


class DiagnosticsPlotterEx(DiagnosticsPlotter):
    LOSSES = {
        **LOSSES_DEFAULT,
        'transformer_train.TOTAL': ('-', 2.0, 'green', "AE training loss (total)"),
        'transformer_valid.TOTAL': ('-', 2.0, 'limegreen', "AE validation loss (total)"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config: CombinedConfigBase = self.context.load_config()

        losses = self.__class__.LOSSES.copy()
        for k in range(len(config.micro.multiresolution)):
            losses.update(_losses(k))
        self.LOSSES = losses


def _losses(k: int):
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    train = ['green', 'limegreen', 'forestgreen'][k]
    valid = ['green', 'limegreen', 'forestgreen'][k]
    return {
        # (linestyle, linewidth, color, label)
        f'transformer_train.layers.layer{k}.v': ('--', None, train, f"AE #{k} t.l. (velocity)"),
        f'transformer_train.layers.layer{k}.vorticity': ('-.', None, train, f"AE #{k} t.l. (vorticity)"),
        f'transformer_train.layers.layer{k}.derivatives': (':', None, train, f"AE #{k} t.l. (derivatives)"),
        f'transformer_train.layers.layer{k}.divergence': (':', None, train, f"AE #{k} t.l. (divergence)"),
        f'transformer_train.layers.layer{k}.pressure': (':', None, train, f"AE #{k} t.l. (pressure)"),
        f'transformer_valid.layers.layer{k}.v': ('--', 0.8, valid, f"AE #{k} v.l. (velocity)"),
        f'transformer_valid.layers.layer{k}.vorticity': ('-.', 0.8, valid, f"AE #{k} v.l. (vorticity)"),
        f'transformer_valid.layers.layer{k}.derivatives': (':', 0.8, valid, f"AE #{k} v.l. (derivatives)"),
        f'transformer_valid.layers.layer{k}.divergence': (':', 0.8, valid, f"AE #{k} v.l. (divergence)"),
        f'transformer_valid.layers.layer{k}.pressure': (':', 0.8, valid, f"AE #{k} v.l. (pressure)"),
    }


class LayerTracerEx(LayerTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: Optional[CombinedConfigBase] = None

    def process_layer_output(self, x):
        # Split velocity, optional vorticity and optional pressure when
        # visualizing layer histograms.

        # # Here we don't really know if we are processing micro or macro state,
        # # i.e. whether we are at the beginning of the encoder or at the end of
        # # the decoder. Hence we must guess the structure from the number of
        # # channels.
        config = self.config
        if not isinstance(x, TensorCollection) and x.ndim == 2 + len(config.micro.cells):
            num_channels = x.shape[1]
            if ('I' in self.prefix or 'E' in self.prefix) \
                    and len(self.prefix) >= 3 \
                    and self.prefix[-2] == 'layers' \
                    and self.prefix[-1].startswith('layer') \
                    and num_channels == config.micro.get_num_export_channels():
                micro_helper = MicroStateHelper(config.micro)
                parts = micro_helper.layer_to_collection(-1, x, no_compute=True)
                super().process_layer_output(parts)
                return
        super().process_layer_output(x)


class MergedRecordsPlotterEx(MergedRecordsPlotter):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument_group('cubismup-report').add_argument
        add('--report-ref-execution-time', type=str, default='raw_client_benchmark.csv',
            help=".csv files with reference (no adaled) execution times")
        add('--report-ref-extension', type=int, default=0,
            help="fill too short ref runs by repeating last --report-ref-extension steps")
        add('--path-suffix', type=str, default='',
            help="to add to output file paths (not implemented everywhere)")

    def record_load_filter(self, keys: Tuple[str, ...], d):
        if keys == ('fields', 'simulations', 'validation', 'cmp_error', 'v'):
            return d
        elif keys == ('fields', 'simulations', 'uncertainty'):
            return d
        else:
            return super().record_load_filter(keys, d)

    def tasks_report_execution_time(self):
        """Plot small data like execution time, stage etc in merged plots."""
        try:
            ref_data: TensorCollection = adaled.load(self.args.report_ref_execution_time)
            ref_data.describe()
            ref_data: np.ndarray = ref_data['execution_time']
        except FileNotFoundError:
            ref_data = None

        try:
            exec_time = parse_execution_time_from_stderr('.')
        except FileNotFoundError:
            exec_time = None

        for key, record in self.records.items():
            yield Task(plot_report_execution_time,
                       f'report-{key}-execution-time-details.png',
                       record['fields', 'metadata'], ref=ref_data,
                       ref_extension=self.args.report_ref_extension,
                       exec_time=exec_time, details=True)
            yield Task(plot_report_execution_time,
                       f'report-{key}-execution-time.png:pdf',
                       record['fields', 'metadata'], ref=ref_data,
                       ref_extension=self.args.report_ref_extension,
                       details=False)

    def tasks_report_validation_error(self):
        suffix = self.args.path_suffix
        for key, record in self.records.items():
            try:
                cmp_error_v = record['fields', 'simulations', 'validation', 'cmp_error', 'v']
            except:
                print("Warning: fields/simulations/validation/cmp_error/v not found, skipping")
                continue
            config: CombinedConfigBase = self.context.load_config()
            yield Task(plot_report_validation_error,
                       f'report-{key}-validation-error-v{suffix}.png:pdf',
                       record['fields', 'metadata'],
                       cmp_error_v[:, 0],
                       config.led.criteria.max_cmp_error)
            yield Task(plot_report_validation_error_detail,
                       f'report-{key}-validation-error-uncertainty{suffix}.png:pdf',
                       config,
                       record['fields', 'metadata'],
                       cmp_error_v[:, 0],
                       uncertainty=record['fields', 'simulations', 'uncertainty', :, 0])

    def plot_report_utilization_and_F(self, *args, F_title: str = "Re", **kwargs):
        plot = super().plot_report_utilization_and_F(*args, F_title=F_title, **kwargs)
        # plot.ax1.set_ylim(450, 1050)
        plot.ax1.yaxis.set_ticks([500, 750, 1000])
        return plot


class TransformerPlotterExBase(TransformerPlotter):
    layer_tracer_cls = LayerTracerEx

    def set_up(self):
        super().set_up()
        self.config: CombinedConfigBase = self.context.load_config()
        self.layer_tracer.config = self.config

    def network_for_histograms(self):
        network = super().network_for_histograms()
        network['PP'] = AutoencoderReconstructionLayer(self.config.micro)  # Postproduction.
        return network

    def postprocess_value_histograms(self, layers):
        for layer in layers:
            layer.name = layer.name.replace("layers.layer", "layer")
        return layers


class TransformerPlotterEx2D(TransformerPlotterExBase):
    state_batch_size = 1
    default_num_states = 50
    default_num_trajectories = 2


class TransformerPlotterEx3D(TransformerPlotterExBase):
    state_batch_size = 4
    default_num_states = 100
    default_num_trajectories = 10


PLOTTERS_2D = [DatasetPlotterEx, DiagnosticsPlotterEx, TransformerPlotterEx2D]
PLOTTERS_3D = [DatasetPlotterEx, DiagnosticsPlotterEx, TransformerPlotterEx3D]
