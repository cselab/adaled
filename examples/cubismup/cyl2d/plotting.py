#!/usr/bin/env python3

# Preload, because it uses a newer version of OpenMP compared to torch.
import cubismup2d

from adaled.led import AdaLEDStage
from adaled.plotting.base import Task, Plotter
from adaled.plotting.plots import Axes, Plot, mpl, plt
from adaled.plotting.utils import fade_color, get_default_rgba
from adaled.plotting.plots_multiresolution import plot_multiresolution_2d
from adaled.postprocessing.record import load_and_concat_records
from ..common.plotting import PLOTTERS_2D, CUPPlotter2D, \
        MergedRecordsPlotterEx, PlotMultiresolution1D, \
        plot_report_validation_error
from ..common.postprocess import find_records, load_cup_forces
from .micro import CylinderMicroConfig, CUP2DFlowBehindCylinderSolver
from .postprocess_utils import extend_runtime_postprocess_data
from .setup import Config
import adaled

import numpy as np
import torch

from typing import List, Optional, Sequence, Tuple
import argparse
import glob
import warnings

def compute_Cd(config: Config, ts: np.ndarray, forces: np.ndarray, rho: float = 1.0):
    """Compute either pressure and viscous drag or both together, depending on
    the shape of `forces`."""
    Re_func = config.make_F_func()
    t = ts * config.micro.dt_macro
    u_inf = config.micro.Re_to_vel(Re_func(t))
    u_inf_norm = np.sqrt((u_inf ** 2).sum(axis=-1))
    factor = 2 / (rho * u_inf_norm ** 3 * (2 * config.micro.r))

    if forces.shape[1] == 2:
        Cd = factor * (forces * u_inf).sum(axis=-1)
        return Cd
    elif forces.shape[1] == 4:
        Cd_p = factor * (forces[:, 0:2] * u_inf).sum(axis=-1)
        Cd_v = factor * (forces[:, 2:4] * u_inf).sum(axis=-1)
        return Cd_p, Cd_v
    else:
        raise ValueError(f"unexpected shape {forces.shape}")


def plot_multiresolution_2d_ex(path: str, config: CylinderMicroConfig, report: bool = False):
    mr = config.make_multiresolution()
    kwargs = {}
    if report:
        kwargs['figsize'] = (6, 3)
    with Plot(path, **kwargs) as (fig, ax):
        ax: Axes
        plot_multiresolution_2d(ax, mr, report=report)

        scale = config.cells[0]
        xy = (scale * config.center[0], scale * config.center[1])
        plt.rc('hatch', linewidth=2)
        # https://stackoverflow.com/questions/5195466/matplotlib-does-not-display-hatching-when-rendering-to-pdf
        ax.add_patch(mpl.patches.Circle(
                xy, scale * config.r,
                edgecolor='none', alpha=0.15, facecolor='black'))
        ax.add_patch(mpl.patches.Circle(
                xy, scale * config.r,
                edgecolor='black', alpha=0.15, facecolor='none', hatch='////'))
        # ax.axvline(xy[0], linestyle=':', linewidth=1.0, color='gray')


def plot_multiresolution_1d(path: str, config: CylinderMicroConfig, d: int, report=True):
    kwargs = {}
    if report:
        kwargs['figsize'] = (6, 1.5)
    with PlotMultiresolution1D(path, config, d, **kwargs) as (plot, fig, ax):
        scale = max(config.micro.cells)
        x0 = scale * (config.micro.center[d] - config.micro.r)
        x1 = scale * (config.micro.center[d] + config.micro.r)
        plot.plot_hatched_obstacle(x0, x1)


def plot_forces_and_drag(
        path: str,
        postprocessed_data: adaled.TensorCollection,
        config: Config,
        total: bool,
        forces: bool = True,
        drag: bool = False,
        report: bool = False,
        timestep_range: Tuple[int, int] = (0, np.inf),
        record_timesteps: Optional[np.ndarray] = None,
        record_x_forces: Optional[np.ndarray] = None):
    timestep = postprocessed_data['metadata', 'timestep']
    lo = max(0, np.searchsorted(timestep, timestep_range[0]) - 1)
    hi = np.searchsorted(timestep, timestep_range[1]) + 2
    timestep = timestep[lo:hi]
    led_forces = postprocessed_data['postprocess', 'manual_forces', lo:hi]
    has_pressure = config.micro.pressure_in_state

    all_forces = [
        (timestep, led_forces['micro'], "micro", '--', 0.0),
        (timestep, led_forces['macro'], "macro", '-.', 0.5),
    ]

    try:
        dt = config.micro.dt_macro
        cup = load_cup_forces(min_t=max(dt * timestep_range[0], timestep[0]),
                              max_t=min(dt * timestep_range[1], timestep[-1]))
        all_forces.append((cup['t'] / dt, cup['FPresVisc'], "CUP", ':', 0.0))
    except FileNotFoundError:
        warnings.warn("cubismup/forceValues_0.dat not found, not plotting CUP forces")
        cup =  None

    assert (record_x_forces is not None) == bool(config.micro.forces_in_state)
    if record_x_forces is not None:
        lo_r = max(0, np.searchsorted(record_timesteps, timestep_range[0]) - 1)
        hi_r = np.searchsorted(record_timesteps, timestep_range[1]) + 2
        all_forces.append((record_timesteps[lo_r:hi_r],
                            record_x_forces[lo_r:hi_r], "CUP record", '-', 0.0))

    kwargs = {'figsize': (10.0, 2.0)} if report else {}
    plot = Plot(path, **kwargs)
    ax = plot.ax

    main_colors = get_default_rgba(4)
    min_ts = +np.inf
    max_ts = -np.inf
    min_y = +np.inf
    max_y = -np.inf

    def _plot_F_pv(ts, F, name, linestyle, fade, colors, kwargs):
        assert F.shape[1] == 4, F.shape
        nonlocal min_y, max_y

        if forces:
            if not np.isnan(F[:, 0]).all():
                ax.plot(ts, F[:, 0], color=colors[0], label=f"$F_x^{{pressure}}$ ({name})", **kwargs)
                ax.plot(ts, F[:, 1], color=colors[1], label=f"$F_y^{{pressure}}$ ({name})", **kwargs)
            ax.plot(ts, F[:, 2], color=colors[2], label=f"$F_x^{{viscous}}$ ({name})", **kwargs)
            ax.plot(ts, F[:, 3], color=colors[3], label=f"$F_y^{{viscous}}$ ({name})", **kwargs)
        if drag:
            Cd_p, Cd_v = compute_Cd(config, ts, F)
            ax.plot(ts, Cd_p, color='black', label=f"Cd_p ({name})", **kwargs)
            ax.plot(ts, Cd_v, color='gray', label=f"Cd_v ({name})", **kwargs)
            min_y = min(min_y, np.nanmin(Cd_p[10:]), np.nanmin(Cd_v[10:]))
            max_y = max(max_y, np.nanmax(Cd_p[10:]), np.nanmax(Cd_v[10:]))

    def _plot_F_total(ts, F, name, linestyle, fade, colors, kwargs):
        assert F.shape[1] == 2, F.shape
        nonlocal min_y, max_y

        if forces:
            ax.plot(ts, F[:, 0], color=colors[0], label=f"$F_x^{{total}}$ ({name})", **kwargs)
            ax.plot(ts, F[:, 1], color=colors[1], label=f"$F_y^{{total}}$ ({name})", **kwargs)
        if drag:
            Cd = compute_Cd(config, ts, F)
            ax.plot(ts, Cd, color='black', label=f"Cd ({name})", **kwargs)
            min_y = min(min_y, np.nanmin(Cd[10:]))
            max_y = max(max_y, np.nanmax(Cd[10:]))

    for ts, F, name, linestyle, fade in all_forces:
        kwargs = {'linestyle': linestyle, 'linewidth': 1.0}
        colors = [fade_color(color, fade) for color in main_colors]
        if total:
            if F.shape[1] == 4:
                F = F[:, 0:2] + F[:, 2:4]
            _plot_F_total(ts, F, name, linestyle, fade, colors, kwargs)
        elif F.shape[1] == 4:
            _plot_F_pv(ts, F, name, linestyle, fade, colors, kwargs)

        min_ts = min(min_ts, ts[0])
        max_ts = max(max_ts, ts[-1])
        if forces:
            min_y = min(min_y, np.nanmin(F[10:]))
            max_y = max(max_y, np.nanmax(F[10:]))

    ax.set_xlim(min_ts, max_ts)
    ax.set_xlabel("timestep")

    if total:
        if drag:
            ax.set_ylabel("total drag")
        else:
            ax.set_ylabel("total forces")
    else:
        if drag:
            ax.set_ylabel("pressure and viscous drag")
        else:
            ax.set_ylabel("pressure and viscous forces")
    ax.set_ylim(min_y - (max_y - min_y) * 0.1,
                max_y + (max_y - min_y) * 0.1)
    ax.grid()

    if not report:
        ax.legend()
    return plot


def plot_forces_error(
        path: str,
        metadata: adaled.TensorCollection,
        x_forces: np.ndarray,
        z_forces: np.ndarray,
        labels: Sequence[str]):
    plot = Plot(path)
    ax = plot.ax

    t = metadata['timestep']
    is_macro = metadata['stage'] == AdaLEDStage.MACRO
    colors = get_default_rgba(len(x_forces.T))
    for color, x_force, label in zip(colors, x_forces.T, labels):
        # ax.plot(t, x_force, linewidth=0.5, color=fade_color(color, 0.7),
        #         label=label + " (expected)")
        ax.plot(t, x_force, linewidth=2, color=color,
                label=label + " (expected)")
    for color, x_force, z_force, label in zip(colors, x_forces.T, z_forces.T, labels):
        z_force = np.ma.array(z_force, mask=~is_macro)
        # ax.plot(t, x_force - z_force, linewidth=0.5, color=color,
        #         label=label + " (error)")
        ax.plot(t, x_force - z_force, linewidth=2, color=color, linestyle='--',
                label=label + " (error)")

    y_scale = np.concatenate([x_forces[100:].ravel(), z_forces[100:].ravel()])
    ax.set_xlim(t[0], round(t[-1], -1))
    if not np.isnan(y_scale).all():
        ymin = np.nanmin(y_scale)
        ymax = np.nanmax(y_scale)
        ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
    ax.grid()
    ax.legend()

    return plot



class CylPlotter(CUPPlotter2D):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument_group('cyl2d').add_argument
        add('--drag-timestep-range', type=int, nargs=2, default=(0, np.inf))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: Config = self.context.load_config()
        self._small_records = None

    def load_small_record_fields(self):
        if self._small_records is not None:
            return self._small_records
        pattern =r'/fields/(metadata|simulations/(x_micro_qoi|z)/)'
        self._small_records = out = {
            sim_id: load_and_concat_records(paths, regex=pattern)['fields']
            for sim_id, paths in find_records().items()
        }
        return out

    def tasks_multiresolution_2d(self):
        yield Task(plot_multiresolution_2d_ex, 'multiresolution.png', self.config.micro)

    def tasks_multiresolution_1d(self):
        if len(self.config.micro.multiresolution) == 1:
            return
        yield Task(plot_multiresolution_1d, 'multiresolution-cross-section-x.png', self.config, d=0)
        yield Task(plot_multiresolution_1d, 'multiresolution-cross-section-y.png', self.config, d=1)

    def tasks_report_multiresolution_2d(self):
        yield Task(plot_multiresolution_2d_ex, 'report-multiresolution.png:pdf',
                   self.config.micro, report=True)

    def tasks_report_multiresolution_1d(self):
        if len(self.config.micro.multiresolution) == 1:
            return
        yield Task(plot_multiresolution_1d, 'report-multiresolution-cross-section-x.png:pdf',
                   self.config, d=0, report=True)
        yield Task(plot_multiresolution_1d, 'report-multiresolution-cross-section-y.png:pdf',
                   self.config, d=1, report=True)

    def tasks_forces_and_drag(self):
        for sim_id, data in self.load_postprocessed_or_skip():
            paths = sorted(glob.glob(f'record-{sim_id:03d}-0*.h5'))
            if self.config.micro.forces_in_state:
                record = load_and_concat_records(
                        paths, regex=r'/?fields/(metadata/timestep|simulations/x_micro_qoi)')
                timesteps = record['fields', 'metadata', 'timestep']
                x_forces = record['fields', 'simulations', 'x_micro_qoi', 'cyl_forces', :, 0, :]
            else:
                timesteps = x_forces = None
            kwargs = {
                'postprocessed_data': data,
                'config': self.config,
                'timestep_range': self.args.drag_timestep_range,
                'record_timesteps': timesteps,
                'record_x_forces': x_forces,
            }
            yield Task(plot_forces_and_drag, f'postprocess-forces-{sim_id:03d}.png',
                       total=False, forces=True, drag=False, **kwargs)
            yield Task(plot_forces_and_drag, f'report-drag-{sim_id:03d}.png',
                       total=False, forces=False, drag=True, report=True, **kwargs)
            yield Task(plot_forces_and_drag, f'postprocess-forces-total-{sim_id:03d}.png',
                       total=True, forces=True, drag=False, **kwargs)
            yield Task(plot_forces_and_drag, f'report-drag-total-{sim_id:03d}.png',
                       total=True, forces=False, drag=True, report=True, **kwargs)

    def tasks_forces_error(self):
        if not self.config.micro.forces_in_state:
            return
        transformer: adaled.CompoundAutoencoder = \
                self.context.load_transformer('transformer-latest.pt')
        for sim_id, fields in self.load_small_record_fields().items():
            sim = fields['simulations']
            metadata = fields['metadata']
            z_macro = sim['z', 'macro']
            assert z_macro.shape[1] == 1  # Simulation batch.
            z_forces = transformer.partial_inverse_transform(
                    torch.from_numpy(z_macro[:, 0, :]), 'cyl_forces').numpy()
            assert z_forces.ndim == 2
            x_forces = sim['x_micro_qoi', 'cyl_forces', :, 0]

            assert len(metadata) == len(x_forces) == len(z_forces), \
                   f"unsupported `every`: {len(metadata)}, {len(x_forces)}, {len(z_forces)}"
            if self.args.record_slice:
                s = slice(*self.args.record_slice)
                metadata = metadata[s]
                x_forces = x_forces[s]
                z_forces = z_forces[s]

            if x_forces.shape[1] == 4:
                yield Task(plot_forces_error, f'postprocess-Fp-error-{sim_id:03d}.png',
                           metadata=metadata,
                           x_forces=x_forces[:, 0:2],
                           z_forces=z_forces[:, 0:2],
                           labels=["$F^{pressure}_x$", "$F^{pressure}_y$"])
                yield Task(plot_forces_error, f'postprocess-Fv-error-{sim_id:03d}.png',
                           metadata=metadata,
                           x_forces=x_forces[:, 2:4],
                           z_forces=z_forces[:, 2:4],
                           labels=["$F^{viscous}_x$", "$F^{viscous}_y$"])
            else:
                assert x_forces.shape[1] == 2, x_forces.shape
                yield Task(plot_forces_error, f'postprocess-Ftotal-error-{sim_id:03d}.png',
                           metadata=metadata,
                           x_forces=x_forces,
                           z_forces=z_forces,
                           labels=["$F^{total}_x$", "$F^{total}_y$"])

    def tasks_validation_error_norm_E_Fcyl(self):
        if not self.config.micro.forces_in_state:
            return
        for key, post in self.load_postprocessed_runtime_or_skip():
            post = extend_runtime_postprocess_data(post)
            qoi = post['micro_state_qoi']
            mean = ((qoi ** 2).sum(axis=-1) ** 0.5).mean(axis=0)
            for k, v in mean.concat_flatten().items():
                print(f"Average micro RMSE {k}={v}")
            yield Task(plot_report_validation_error,
                       f'report-{key:03d}-validation-error-EFcyl.png:pdf',
                       post['metadata'],
                       post['__qoi_error_norm_Fcyl', :, 0],
                       ylabel="error $E_F$",
                       ylim=(1e-3, None))


class MergedRecordsPlotterEx2(MergedRecordsPlotterEx):
    def record_load_filter(self, keys: Tuple[str, ...], d):
        if keys[:3] == ('fields', 'simulations', 'z'):
            return d
        else:
            return super().record_load_filter(keys, d)

    def plot_report_efficiency_and_F(self, *args, **kwargs):
        plot = super().plot_report_efficiency_and_F(
                *args, F_title=r"Re", macro_smoothing=500, **kwargs)
        config: Config = self.context.load_config()
        max_eff = config.led.criteria.expected_max_utilization(),
        ax1: Axes = plot.ax1
        ax2: Axes = plot.ax2
        ax1.set_ylim(450, 1050)
        ax2.axhline(max_eff, color='limegreen', linewidth=1.0)
        ax2.set_ylim(0.0, 1.0)
        return plot


if __name__ == '__main__':
    from adaled.plotting.all import main
    main([*PLOTTERS_2D, CylPlotter, MergedRecordsPlotterEx2])
