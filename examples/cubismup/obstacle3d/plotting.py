#!/usr/bin/env python3

# Preload, because it uses a newer version of OpenMP compared to torch.
import libcubismup3d

from adaled.plotting.base import Task, Plotter
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.plotting.plots_multiresolution import plot_multiresolution_2d
from ..common.plotting import PLOTTERS_3D, CUPPlotter3D, PlotMultiresolution1D
from .micro import ObstacleMicroConfig
from .setup import Config

import numpy as np

from typing import Tuple
import warnings

def add_obstacle_1d(plot: PlotMultiresolution1D, config: ObstacleMicroConfig, dim: int):
    if config.kind == 'Sphere':
        scale = max(config.cells)
        x0 = scale * (config.center[dim] - 0.5 * config.L)
        x1 = scale * (config.center[dim] + 0.5 * config.L)
        plot.plot_hatched_obstacle(x0, x1)
    else:
        warnings.warn(f"plotting obstacle {config.kind!r} not supported")


def add_obstacle_2d(ax: Axes, config: ObstacleMicroConfig, dims: Tuple[int, int]):
    """Plot obstacle on a multiresolution plot."""
    if config.kind == 'Sphere':
        scale = max(config.cells)
        xy = (scale * config.center[dims[0]], scale * config.center[dims[1]])
        ax.add_patch(mpl.patches.Circle(
                xy, scale * config.L / 2, edgecolor='black', facecolor='none'))
    else:
        warnings.warn(f"plotting obstacle {config.kind!r} not supported")


def plot_multiresolution_1d(path: str, config: Config, dim: int):
    with PlotMultiresolution1D(path, config, dim) as (plot, fig, ax):
        ax: Axes
        add_obstacle_1d(plot, config.micro, dim)
        ax.locator_params(nbins=config.micro.cells[dim], axis='x')
        ax.grid(axis='x', linewidth=0.3, color='lightgray')


def plot_multiresolution_2d_ex(path: str, config: ObstacleMicroConfig, dims: Tuple[int, int]):
    mr = config.make_multiresolution()
    with Plot(path) as (fig, ax):
        ax: Axes
        plot_multiresolution_2d(ax, mr, dims=dims)
        ax.set_xlabel("xyz"[dims[0]])
        ax.set_ylabel("xyz"[dims[1]])

        add_obstacle_2d(ax, config, dims)


class ObstacleCasePlotter(Plotter):
    def tasks_multiresolution_1d(self):
        config: Config = self.context.load_config()
        yield Task(plot_multiresolution_1d, 'multiresolution_1d_x.pdf', config, 0)
        yield Task(plot_multiresolution_1d, 'multiresolution_1d_y.pdf', config, 1)
        yield Task(plot_multiresolution_1d, 'multiresolution_1d_z.pdf', config, 2)

    def tasks_multiresolution_2d(self):
        config: Config = self.context.load_config()
        yield Task(plot_multiresolution_2d_ex, 'multiresolution_xy.png', config.micro, (0, 1))
        yield Task(plot_multiresolution_2d_ex, 'multiresolution_xz.png', config.micro, (0, 2))
        yield Task(plot_multiresolution_2d_ex, 'multiresolution_yz.png', config.micro, (1, 2))


if __name__ == '__main__':
    from adaled.plotting.all import main
    main(PLOTTERS_3D + [CUPPlotter3D, ObstacleCasePlotter])
