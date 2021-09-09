"""Visualization of the network."""

from adaled.backends import TensorCollection
from adaled.plotting.plots import Axes, Plot
from adaled.plotting.utils import human_formatter
from adaled.nn.tracer import Tracer
import adaled
import adaled.utils.io_ as io_

import numpy as np
import torch

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import contextlib
import os

@dataclass
class _HistogramData:
    name: str
    min: float = +np.inf
    max: float = -np.inf
    histogram: Optional[np.ndarray] = None
    bin_x: Optional[np.ndarray] = None


class LayerTracer(Tracer):
    MODE_MINMAX = 1
    MODE_HISTOGRAM = 2

    def __init__(self, num_bins: int = 100):
        self.idx = 0
        self.layer_data: List[_HistogramData] = []
        self.prefix = []
        self.mode: int = self.MODE_MINMAX
        self.num_bins = num_bins

    def reset(self, mode: int):
        self.mode = mode
        self.idx = 0

    @contextlib.contextmanager
    def __call__(self, key):
        if isinstance(key, torch.nn.Module):
            key = getattr(key, 'LAYER_SHORT_NAME', key.__class__.__name__)
        self.prefix.append(str(key))
        try:
            yield None
        finally:
            self.prefix.pop()

    def evaluate(self, model, x):
        if isinstance(model, torch.nn.Sequential):
            for name, child in model.named_children():
                with self(name):
                    x = self.evaluate(child, x)
            return x

        # Some models define __iter__, while in principle they could inherit
        # from torch.nn.Sequential.
        try:
            iter(model)
        except:
            pass
        else:
            for i, child in enumerate(model):
                with self(i):
                    x = self.evaluate(child, x)
            return x

        x = model(x)
        with self(model):
            self.process_layer_output(x)
        return x

    def process_layer_output(self, x):
        if isinstance(x, (dict, TensorCollection)):
            for key, value in x.items():
                with self(key):
                    self.process_layer_output(value)
            return

        y = x.ravel()
        if len(y) == 0:
            return
        if self.idx >= len(self.layer_data):
            self.layer_data.append(_HistogramData(name=".".join(self.prefix)))
        y = y.cpu().numpy().astype(np.float32)
        data = self.layer_data[self.idx]
        if self.mode == self.MODE_MINMAX:
            data.min = min(data.min, y.min())
            data.max = max(data.max, y.max())
        else:
            y = adaled.to_numpy(y)
            histogram, bin_x = np.histogram(
                    y, self.num_bins, range=(data.min, data.max))
            if data.histogram is None:
                data.histogram = histogram
                data.bin_x = bin_x
            else:
                data.histogram += histogram

        self.idx += 1


def compute_layer_value_histograms(
        dataloader: Iterable[Any],
        network: torch.nn.Module,
        tracer: LayerTracer) -> List[_HistogramData]:
    """Evaluate dataset on a network and return a list of histograms, one for
    each layer. Tensor collections are expanded, each having its own item."""
    num_batches = len(dataloader)
    every = (num_batches + 15) // 16

    print(f"Computing value histograms for a dataset of {num_batches} batches...")
    for mode in [tracer.MODE_MINMAX, tracer.MODE_HISTOGRAM]:
        with tracer:
            with torch.no_grad():
                for batch_id, batch in enumerate(dataloader):
                    if batch_id % every == 0:
                        print(f"    mode={mode} batch={batch_id}/{num_batches} batch_size={len(batch)}")
                    tracer.reset(mode)
                    tracer.evaluate(network, batch)

    print("Number of samples per layer:")
    for layer in tracer.layer_data:
        print(f"    {layer.name}: {layer.histogram.sum()}")
    return tracer.layer_data


def plot_histograms(
        output_path: str,
        output_path_log: str,
        histograms: List[_HistogramData]):
    nrows = int(np.ceil(np.sqrt(len(histograms))))
    ncols = (len(histograms) + nrows - 1) // nrows

    print(f"Plotting {len(histograms)} histograms...")
    formatter = human_formatter(digits=0)
    kwargs = dict(nrows=nrows, ncols=ncols, squeeze=False, figsize=(20.0, 12.0))
    with Plot(output_path, **kwargs) as (fig1, axs1), \
         Plot(output_path_log, **kwargs) as (fig2, axs2):
        axs1 = axs1.ravel()
        axs2 = axs2.ravel()
        for ax1, ax2, histo in zip(axs1, axs2, histograms):
            ax1: Axes
            ax2: Axes
            ax1.set_title(histo.name, fontsize=10)
            ax2.set_title(histo.name, fontsize=10)
            ax2.set_yscale('log')
            ax1.stairs(histo.histogram, histo.bin_x, fill=True)
            ax2.stairs(histo.histogram, histo.bin_x, fill=True)
            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax1.ticklabel_format(axis='x', style='plain', scilimits=(-8, 9))
            ax2.ticklabel_format(axis='x', style='plain', scilimits=(-8, 9))
            ax1.yaxis.set_major_formatter(formatter)

        for ax1, ax2 in zip(axs1[len(histograms):], axs2[len(histograms):]):
            fig1.delaxes(ax1)
            fig2.delaxes(ax2)

        fig1.tight_layout()
        fig2.tight_layout()

    path = os.path.splitext(output_path)[0] + '.pt'
    io_.save(histograms, path)
    print(f"Saved histogram data to {path}.")
