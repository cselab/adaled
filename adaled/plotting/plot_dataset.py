#!/usr/bin/env python3

from adaled.plotting.base import Plotter, Task
from adaled.plotting.plots import Plot
import adaled
from adaled.plotting.utils import add_faded_line

import numpy as np

from typing import Sequence
import argparse

def plot_phase_space(
        path: str,
        title: str,
        dataset: Sequence['arraylike']):

    alpha = [np.linspace(0.2, 1.0, len(traj)) for traj in dataset]
    with Plot(path, title=title) as (fig, ax):
        for i, traj in enumerate(dataset):
            add_faded_line(ax, traj[:, 0], traj[:, 1], 0.8,
                           color=np.random.uniform(0.0, 1.0, 3))
        ax.autoscale()


class DatasetPlotter(Plotter):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument_group('dataset').add_argument
        add('--dataset-path', type=str, default='dataset-latest/train')

    def set_up(self):
        if hasattr(self, 'dataset'):
            if self.dataset is None:
                self.skip("dataset not found")
            return
        try:
            self.dataset: adaled.DynamicTrajectoryDataset = \
                    self.context.load_dataset(self.args.dataset_path)
        except FileNotFoundError:
            self.dataset = None
            self.skip("dataset not found")

    def tasks_xy_phase_space(self):
        states: adaled.TensorCollection = self.dataset.as_states()
        if len(states) == 0:
            return
        states = states[0]

        for key, states_ in states.allitems():
            if states_.shape != (2,):
                continue
            suffix = '-'.join(key)
            title_suffix = "/".join(key)

            trajectories = np.asarray(self.dataset.as_trajectories(key))
            yield Task(plot_phase_space,
                       f'plot-dataset-phase-space-{suffix}.png',
                       f"Phase space for {title_suffix}",
                       trajectories)

            trajectories = trajectories[np.random.choice(len(trajectories), 50)]
            yield Task(plot_phase_space,
                       f'plot-dataset-phase-space-{suffix}-subset.png',
                       f"Phase space for {title_suffix}",
                       trajectories)
