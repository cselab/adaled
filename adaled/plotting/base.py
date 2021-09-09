from adaled.led.diagnostics import AdaLEDDiagnostics
from adaled.plotting.tasks import Task, TaskCase, TaskSuite
from adaled.utils.data.datasets import load_dataset
import adaled.utils.io_ as io_

from dataclasses import dataclass, field
from typing import List, Optional
import argparse

class PlottingContext:
    def __init__(self,
                 xlabels: Optional[List[str]] = None,
                 zlabels: Optional[List[str]] = None):
        self.xlabels = xlabels
        self.zlabels = zlabels
        self.xlabels_reconstructed = None

        self._x_shape = None
        self._z_shape = None
        self._cache = {}

    def init(self, x_shape, z_shape):
        # Assert that different invocations of init() use consistent shapes.
        if self._x_shape is not None:
            assert self._x_shape == x_shape, (self._x_shape, x_shape)
            assert self._z_shape == z_shape, (self._z_shape, z_shape)
            return
        self._x_shape = x_shape
        self._z_shape = z_shape

        if self.xlabels is None and isinstance(x_shape, tuple):
            # This will fail for TensorCollection, which is ok.
            self.xlabels = [f"$x_{{{i}}}$" for i in range(x_shape[0])]
        if self.zlabels is None:
            self.zlabels = [f"$z_{{{i}}}$" for i in range(z_shape[0])]

    def load_config(self, path: str = 'config.pt'):
        if path not in self._cache:
            self._cache[path] = io_.load(path)
        return self._cache[path]

    # Multiple functions since the load/save protocol is not yet fully
    # determined (maybe io_.load will is not enough for some objects).
    def load_dataset(self, path, comm=None):
        if path not in self._cache:
            print(f"Loading dataset {path}")
            self._cache[path] = load_dataset(path, comm)
            print(f"Loaded dataset {path}")
        return self._cache[path]

    def load_macro(self, path):
        if path not in self._cache:
            self._cache[path] = macro = io_.load(path)
            for prop in macro.propagators:
                prop.eval()
        return self._cache[path]

    def load_transformer(self, path):
        if path not in self._cache:
            self._cache[path] = transformer = io_.load(path)
            if hasattr(transformer, 'model'):
                transformer.model.eval()
        return self._cache[path]


class Plotter(TaskCase):
    def __init__(self, args):
        self.args = args
        self.context = PlottingContext()
