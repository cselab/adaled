#!/usr/bin/env python3

"""Visualization of transformers."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adaled import TensorCollection, cmap, to_numpy
from adaled.plotting.base import Plotter, Task
from adaled.plotting.network import \
        LayerTracer, compute_layer_value_histograms, plot_histograms
from adaled.utils.data.dataloaders import default_collate
from adaled.utils.data.datasets import \
        apply_transformation_on_dataset, random_dataset_subset
import adaled.plotting.base as base
import adaled.plotting.plots as plots
import adaled

import cycler  # From matplotlib.
import numpy as np
import torch

from collections import OrderedDict
from typing import List
import argparse


class TransformerPlotter(Plotter):
    MAX_CHANNELS = 5
    default_num_states: int = 50
    default_num_trajectories: int = 5
    state_batch_size: int = 16
    layer_tracer_cls = LayerTracer

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_argument_group('transformer').add_argument
        add('--transformer-path', type=str, default='transformer-latest.pt',
            help="path of the transformer pickle file")
        add('--transformer-dataset-path', type=str,
            default='dataset-latest/valid', help="dataset to use")
        add('--transformer-num-states', type=int,
            default=cls.default_num_states, help="number of states to plot")
        add('--transformer-num-trajectories', type=int,
            default=cls.default_num_trajectories, help="number of trajectories to plot")
        add('--transformer-num-histogram-samples', type=int, default=3000,
            help="number of sample to use for the histogram")

    def set_up(self):
        if getattr(self, 'skip_reason', False):
            self.skip(self.skip_reason)
        if hasattr(self, 'transformer'):
            return
        args = self.args
        path = args.transformer_path
        self.output_prefix = os.path.splitext(path)[0] + '-'
        try:
            self.transformer = self.context.load_transformer(path)
        except FileNotFoundError:
            self.skip_reason = "transformer not found"
            self.skip(self.skip_reason)
        if isinstance(self.transformer, adaled.IdentityTransformer):
            self.skip_reason = "identity transformer"
            self.skip(self.skip_reason)
        try:
            self.dataset = self.context.load_dataset(
                    args.transformer_dataset_path)
        except FileNotFoundError:
            self.skip_reason = "dataset not found"
            self.skip(self.skip_reason)
        states_dataset = self.dataset.as_states('trajectory')
        count = min(len(states_dataset), args.transformer_num_states)
        print(f"Reading {count} states", flush=True)
        self.state_subset = default_collate(random_dataset_subset(
                states_dataset, lazy=False, count=count))

        traj_dataset = self.dataset.as_trajectories('trajectory')
        count = min(len(traj_dataset), args.transformer_num_trajectories)
        print(f"Reading {count} trajectories", flush=True)
        self.trajectory_subset = default_collate(random_dataset_subset(
                traj_dataset, lazy=False, count=count))

        self.layer_tracer = self.layer_tracer_cls()

    def preprocess(self):
        if hasattr(self, 'reconstructed_trajectories'):
            return
        batch_size = 4  # Only affects performance and memory usage.
        self.reconstructed_trajectories = None
        with torch.no_grad(), \
                adaled.set_train(self.transformer.model, False):
            print("Applying transformation on states...")
            x = to_numpy(self.state_subset['x'])
            z = apply_transformation_on_dataset(self.transformer, x, batch_size, verbose=True)
            x_reconstructed = to_numpy(apply_transformation_on_dataset(
                    self.transformer.inverse_transform, z, batch_size, verbose=True))
            # mse = (x - x_reconstructed) ** 2
            # mse = mse.mean(axis=tuple(range(1, x.ndim)))
            # rmse = mse ** 0.5
            # print(f"Sample MSE loss:  avg={mse.mean():.6g}  "
            #       f"min={mse.min():.6}  max={mse.max():.6}")
            # print(f"Sample RMSE loss: avg={rmse.mean():.6g}  "
            #       f"min={rmse.min():.6}  max={rmse.max():.6}")
            self.reconstructed_samples = x_reconstructed

            print("Applying transformation on trajectories...")
            x = to_numpy(self.trajectory_subset['x'])
            z = apply_transformation_on_dataset(
                    self.transformer,
                    cmap(lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:]), x),
                    batch_size, verbose=True)
            x_reconstructed = to_numpy(apply_transformation_on_dataset(
                    self.transformer.inverse_transform, z, batch_size, verbose=True))
            try:
                x_reconstructed = cmap(lambda a_rec, a: a_rec.reshape(a.shape), x_reconstructed, x)
            except ValueError:
                x_reconstructed = None
            # mse = ((x - x_reconstructed) ** 2).mean()
            # print(f"Trajectory MSE loss:  {mse:.6g}")
            # print(f"Trajectory RMSE loss: {mse ** 0.5:.6g}")
            self.reconstructed_trajectories = x_reconstructed

        # Pass the shapes of one x state and one z state.
        self.context.init(x[0, 0].shape, z[0].shape)

    def make_path(self, suffix):
        return self.output_prefix + suffix

    def tasks_sample_reconstruction_as_lines(self):
        x = self.state_subset['x']
        if isinstance(x, TensorCollection) or x.ndim != 2:
            self.skip("only for 1d data")
        self.preprocess()

        yield Task(plots.MatrixScatterPlot,
                   self.make_path('sample-reconstruction-lines.png'),
                   "Dataset samples reconstruction",
                   [x, self.reconstructed_trajectories],
                   ["original", "reconstructed"],
                   # Original is empty circle, reconstructed is full triangles.
                   cycler.cycler(marker=['o', 'v'], facecolors=['none', None]),
                   self.context.xlabels)

    def tasks_sample_reconstruction_channels(self):
        x = self.state_subset['x']
        if isinstance(x, TensorCollection) or x.ndim != 3:
            self.skip("only for 2d data")
        if x.shape[1] > self.MAX_CHANNELS:
            self.skip("too many channels")
        self.preprocess()

        x = to_numpy(x[:8])
        x_reconstructed = to_numpy(self.reconstructed_samples[:8])
        error = x_reconstructed - x
        error_filler = np.full_like(error, np.nan)
        # To skip two colors.
        error = np.stack([error_filler, error_filler, error])

        yield Task(plots.LinePlotMatrix,
                   self.make_path('sample-reconstruction-channels.png'),
                   "Random state samples and their reconstruction",
                   np.stack([x, x_reconstructed]),
                   xlabel="x",
                   channel_names=self.context.xlabels,
                   dataset_names=["real", "reconstructed"],
                   grid=True)
        yield Task(plots.LinePlotMatrix,
                   self.make_path('sample-reconstruction-channels-error.png'),
                   "Absolute reconstruction error of random state samples",
                   error,
                   xlabel="x",
                   channel_names=self.context.xlabels,
                   dataset_names=["", "", "absolute error"],
                   grid=True)

    def tasks_trajectories_reconstruction_channels(self):
        xs = self.trajectory_subset['x']
        if isinstance(xs, TensorCollection) or xs.ndim != 4:
            self.skip("only for 2d data")
        if xs.shape[2] > self.MAX_CHANNELS:
            self.skip("too many channels")
        if self.reconstructed_trajectories is None:
            self.skip("x_micro and x_macro shapes don't match")
        self.preprocess()

        xs = to_numpy(xs)
        xs_reconstructed = to_numpy(self.reconstructed_trajectories)

        xlabels = self.context.xlabels
        for i, (x, x_rec) in enumerate(zip(xs, xs_reconstructed)):
            x = np.moveaxis(x, 1, 0)
            x_rec = np.moveaxis(x_rec, 1, 0)
            labels = [(l, l + " (reconstructed)", l + " (absolute error)")
                      for l in xlabels]
            dx = x_rec - x
            mse = float((dx ** 2).mean())
            yield Task(plots.ImageErrorPlotMatrix,
                       self.make_path(f'trajectory-reconstruction-{i:02d}.png'),
                       f"Random dataset trajectory + reconstruction "
                       f"(MSE={mse:.5g}, RMSE={mse ** 0.5:.5g})",
                       list(zip(x, x_rec)), labels)

    def tasks_histograms(self):
        dataloader = adaled.utils.data.dataloaders.DataLoader(
                adaled.cmap(torch.as_tensor, self.state_subset['x']),
                batch_size=self.state_batch_size)

        network = self.network_for_histograms()
        if isinstance(network, OrderedDict):
            network = torch.nn.Sequential(network)
        data = compute_layer_value_histograms(dataloader, network, self.layer_tracer)
        data = self.postprocess_value_histograms(data)
        yield Task(plot_histograms,
                   self.make_path('layer-histogram.png'),
                   self.make_path('layer-histogram-log.png'),
                   data)

    # virtual
    def network_for_histograms(self) -> OrderedDict:
        return OrderedDict({
            'I': torch.nn.Identity(),
            'E': self.transformer.model.encoder,
            'D': self.transformer.model.decoder,
        })

    # virtual
    def postprocess_value_histograms(self, layers: List['_HistogramData']) \
            -> List['_HistogramData']:
        return layers


if __name__ == '__main__':
    TransformerPlotter.main()
