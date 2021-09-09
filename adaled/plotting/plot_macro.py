#!/usr/bin/env python3

"""Visualization of the macro solver."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adaled.plotting.base import Plotter, Task
from adaled.plotting.plots import Axes
from adaled.utils.arrays import join_sequences
from adaled.utils.data.dataloaders import default_collate
from adaled.utils.data.datasets import random_dataset_subset
import adaled.plotting.base as base
import adaled.plotting.plots as plots
import adaled

import numpy as np
import torch

from typing import Any, Iterable, Sequence, Optional
import argparse


def add_trajectories_plot(
        ax: Axes,
        datasets: Sequence['arraylike'],
        dataset_labels: Sequence[str]):
    for trajectories, label in zip(datasets, dataset_labels):
        # Render all trajectories in one go by joining them with a column of
        # nans in between.
        flat = join_sequences(trajectories, gap=np.nan)
        ax.plot(flat[:, 0], flat[:, 1], label=label)


class MacroTrajectoryPlotter(Plotter):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument_group('transformer').add_argument
        add('--macro-path', type=str, default='macro-latest.pt',
            help="path of the transformer pickle file")
        add('--macro-dataset-path', type=str,
            default='dataset-latest/valid', help="dataset to use")
        add('--macro-num-trajectories', type=int, default=8,
            help="number of trajectories to plot")

    def set_up(self):
        if hasattr(self, 'macro'):
            return
        args = self.args
        self.output_prefix = os.path.splitext(args.macro_path)[0] + '-'
        self.macro: adaled.MacroSolver = self.context.load_macro(args.macro_path)
        self.transformer = self.context.load_transformer(args.transformer_path)

        # Use flags from other plotter. FIXME: this is not ideal design.
        try:
            self.dataset = self.context.load_dataset(args.transformer_dataset_path)
            traj_dataset = self.dataset.as_trajectories('trajectory')
            self.trajectory_subset = default_collate(random_dataset_subset(
                    traj_dataset, lazy=False,
                    count=min(len(traj_dataset), args.macro_num_trajectories)))
        except FileNotFoundError:
            self.dataset = None
            self.trajectory_subset = None
        except TypeError:  # empty
            print("Error loading the dataset, skipping dataset plots.")
            self.dataset = None
            self.trajectory_subset = None

    def make_path(self, suffix):
        return self.output_prefix + suffix

    def tasks_prediction_with_channels(self):
        if self.trajectory_subset is None:
            self.skip("dataset not available")
        forcing_steps = 10

        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)

        trajectories = self.trajectory_subset
        if isinstance(trajectories, adaled.TensorCollection) \
                or len(trajectories.shape) != 5:
            # TODO: document this function, what does it even do?
            self.skip("input not supported")
        for i, x_micro in enumerate(self.trajectory_subset):
            with torch.no_grad():
                z = self.transformer.transform(x_micro[:forcing_steps])
                z = z[None, ...]  # Add batch size of 1.
                z, hidden = self.macro.advance_multiple(
                        z, extra_steps=(len(x_micro) - forcing_steps - 1))
                z = z[0, ...]  # Remove batch dimension.
                x_macro = self.transformer.inverse_transform(z)

            self.context.init(x_macro[0].shape, z[0].shape)

            # TODO: Redesign the plotting code such that trajectory generation
            # logic is split from the rendering.

            # Put channel axis in front.
            x_micro = adaled.to_numpy(x_micro['x'])
            x_macro = adaled.to_numpy(x_macro)
            x_micro = np.moveaxis(x_micro, 1, 0)
            x_macro = np.moveaxis(x_macro, 1, 0)
            assert x_macro.ndim == 3
            labels = [(l, l + " (predicted)", l + " (absolute error)")
                      for l in self.context.xlabels]
            yield Task(plots.ImageErrorPlotMatrix,
                       self.make_path(f'prediction-{i}.png'),
                       "Macro solver prediction: validation trajectory, predicted and error",
                       list(zip(x_micro, x_macro)), labels)

    # TODO: Fix this.
    def __old_tasks(self,
              num_steps: int = 10,
              teacher_forcing_steps: int = 10):
        if teacher_forcing_steps <= 0:
            raise ValueError("teacher_forcing_steps must be at least 1")

        dataset = self.dataset.as_trajectories()
        trajectory_length = len(trajectories[0])

        with torch.no_grad():
            print("Transforming trajectories x -> z.")
            z_ground_truth = self.transformer.transform(trajectories)
            z_prefix = z_ground_truth[:, :teacher_forcing_steps]
            Fs = trajectories['F']
            # Rearrange as (time step, batch, ...).
            Fs = adaled.cmap(lambda f: adaled.get_backend(f).moveaxis(f, 0, 1), Fs)
            print(f"Evaluating macro solver with {teacher_forcing_steps} "
                  f"steps of teacher forcing.")
            z_prediction, h, uncertainties = self.macro.advance_multiple(
                    z_prefix, F_batches=Fs,
                    extra_steps=trajectory_length - teacher_forcing_steps,
                    compute_uncertainty=True)
            print("Done.")

        zlabels = [f"$z_{{{i}}}$" for i in range(z_prefix.shape[-1])]
        yield Task(plots.MultidimensionalStateMatrixPlot,
                   self.make_path(f'prediction-1-step.png'),
                   f"First prediction step vs ground truth (latent space)\n",
                   add_trajectories_plot,
                   datasets=[z_ground_truth[:, :2], z_prediction[:, :2]],
                   dataset_labels=["ground truth", "macro prediction"],
                   axes_labels=zlabels)

        yield Task(plots.MultidimensionalStateMatrixPlot,
                   self.make_path(f'prediction-{num_steps}-steps.png'),
                   f"Prediction vs ground truth (latent space)\n"
                   f"(first {num_steps} steps with {teacher_forcing_steps} "
                   f"steps of teacher forcing)",
                   add_trajectories_plot,
                   datasets=[z_ground_truth[:, :num_steps + 1],
                             z_prediction[:, :num_steps + 1]],
                   # uncertainty_sets=[None, uncertainties],
                   dataset_labels=["ground truth", "macro prediction"],
                   axes_labels=zlabels)

        yield Task(plots.MultidimensionalStateMatrixPlot,
                   self.make_path('prediction-full-trajectory.png'),
                   "Trajectory prediction vs ground truth (in latent space)",
                   add_trajectories_plot,
                   datasets=[z_ground_truth, z_prediction],
                   dataset_labels=["ground truth", "macro prediction"],
                   axes_labels=zlabels)


if __name__ == '__main__':
    MacroTrajectoryPlotter.main()
