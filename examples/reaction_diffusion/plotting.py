#!/usr/bin/env python3

from adaled.plotting.base import Task, Plotter
from adaled.plotting.plots import Plot
from adaled.plotting.plot_record import MergedRecordsPlotter
from examples.reaction_diffusion.setup import Config
import adaled

import numpy as np

from typing import Tuple


def plot_postprocessed(
        path: str,
        config: Config,
        post: adaled.TensorCollection) -> Plot:
    # TODO: Remove offline postprocessing, it was replaced by online validation.
    plot = Plot(path)
    ax = plot.ax

    t = post['metadata', 'timestep']
    is_macro = post['metadata', 'stage'] == adaled.AdaLEDStage.MACRO
    mse = post['errors', 'mse_rel'].copy()
    mse[~is_macro] = np.nan
    ax.axhline(config.criteria.max_cmp_error, t[0], round(t[-1], -2), color='red')
    ax.plot(t, mse, color='red')
    ax.grid()

    ax.set_xlabel("time step")
    ax.set_ylabel("MSE")
    ax.set_yscale('log')
    return plot


def plot_validation(
        path: str,
        config: Config,
        fields: adaled.TensorCollection) -> Plot:
    plot = Plot(path)
    ax = plot.ax

    t = fields['metadata', 'timestep']
    is_macro = fields['metadata', 'stage'] == adaled.AdaLEDStage.MACRO
    val = fields['simulations', 'validation'].clone()
    val[~is_macro] = np.nan
    ax.axhline(config.criteria.max_cmp_error, t[0], round(t[-1], -2), color='red')
    ax.plot(t, val['macro_micro_x_rel'],
            label=r"RelMSE $D(z_{\mathrm{macro}})$ vs $x_{\mathrm{micro}}$")
    ax.plot(t, val['macro_ae_x_rel'],
            label=r"RelMSE $D(z_{\mathrm{macro}})$ vs $D(E(D(z_{\mathrm{macro}})))$")
    ax.plot(t, val['macro_micro_z'], '--',
            label=r"MSE $z_{\mathrm{macro}}$ vs $E(x_{\mathrm{micro}})$")
    ax.plot(t, val['macro_ae_z'], '--',
            label=r"MSE $z_{\mathrm{macro}}$ vs $E(D(z_{\mathrm{macro}}))$")
    ax.grid()
    ax.legend()

    ax.set_xlabel("time step")
    ax.set_ylabel("MSE")
    ax.set_yscale('log')
    return plot


class RDPlotter(Plotter):
    def tasks_last_movie_frame(self):
        from .movie import plot_movie
        yield Task(plot_movie,
                   'movie-last.png',
                   self.context.load_config(),
                   frame_begin=-1)

    def tasks_postprocessing(self):
        try:
            post = adaled.load('postprocessed.pt')
        except FileNotFoundError:
            print("Warning: postprocessed.pt not found, skipping.")
        else:
            yield Task(plot_postprocessed,
                       'postprocessed-error.png',
                       self.context.load_config(),
                       post)


class MergedRecordsPlotterEx(MergedRecordsPlotter):
    def record_load_filter(self, keys: Tuple[str, ...], d):
        if 'validation' in keys:
            return d
        else:
            return super().record_load_filter(keys, d)

    def tasks_validation(self):
        for key, record in self.records.items():
            yield Task(plot_validation,
                       f'validation-{key}.png',
                       self.context.load_config(),
                       record['fields'])



if __name__ == '__main__':
    from adaled.plotting.all import main
    main([RDPlotter, MergedRecordsPlotterEx])
