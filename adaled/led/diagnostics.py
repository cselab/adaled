from adaled.backends import TensorCollection, cmap
from adaled.transformers.base import Transformer
from adaled.utils.buffer import DynamicArray
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.data.collections import DynamicTrajectoryDatasetCollection
from adaled.utils.data.datasets import TrajectoryDataset
from adaled.utils.misc import to_numpy
import adaled
import adaled.utils.io_ as io_

import numpy as np

from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import copy
import dataclasses
import os
import threading
import time

nan64 = np.float64(np.nan)

__all__ = [
    'nan64', 'AdaLEDStage', 'AdaLEDStep', 'AdaLEDCycleDiagnostics',
    'AdaLEDDiagnostics', 'HistogramStatsConfig', 'load_diagnostics',
    'postprocess_diagnostics',
]

class AdaLEDStage(IntEnum):
    """Stage of the AdaLED cycle."""
    WARMUP = 1
    COMPARISON = 2
    MACRO = 3
    MICRO = 4
    RELAXATION = 5

    MASK_BOTH: int
    MASK_ONLY_MICRO: int
    MASK_ONLY_MACRO: int
    MASK_MICRO: int
    MASK_MACRO: int

AdaLEDStage.MASK_BOTH = (1 << AdaLEDStage.WARMUP) | (1 << AdaLEDStage.COMPARISON)
AdaLEDStage.MASK_ONLY_MICRO = (1 << AdaLEDStage.MICRO) | (1 << AdaLEDStage.RELAXATION)
AdaLEDStage.MASK_ONLY_MACRO = 1 << AdaLEDStage.MACRO
AdaLEDStage.MASK_MICRO = AdaLEDStage.MASK_BOTH | AdaLEDStage.MASK_ONLY_MICRO
AdaLEDStage.MASK_MACRO = AdaLEDStage.MASK_BOTH | AdaLEDStage.MASK_ONLY_MACRO


@dataclasses.dataclass
class AdaLEDStep:
    """Data of one AdaLED time step.

    Attributes:
        stage (AdaLEDStage): the cycle stage
        x (optional collection-like):
            the micro state, as returned by micro propagators' ``get_state()``
        z (optional collection-like):
            the macro state, as returned by the macro propagator
        F (optional collection-like):
            external forcing
        transformer (optional Transformer):
            transformer used during the current cycle
        uncertainty (optional collection-like):
            prediction uncertainty, as computed by the macro propagator
        hidden (optional):
            hidden state of the macro propagator
    """
    __slots__ = ('stage', 'x', 'z', 'F', 'transformer', 'uncertainty', 'hidden')
    stage: AdaLEDStage
    x: Optional[np.ndarray]      # Micro state.
    z: Optional['torch.Tensor']  # Latent macro state.
    F: Optional[np.ndarray]      # External forcing.
    transformer: Optional[Transformer]
    uncertainty: Optional[Any]
    hidden: Optional[Any]

    def __str__(self):
        # The default implementation may print annoyingly long xs and zs.
        hidden = "..." if self.hidden is not None else None
        with np.printoptions(linewidth=120, threshold=32, edgeitems=2):
            return f"{self.__class__.__name__}(stage={self.stage}, " \
                   f"x={self.x}, z={self.z}, F={self.F}, " \
                   f"uncertainty={self.uncertainty}, hidden={hidden})"


@dataclasses.dataclass
class AdaLEDCycleDiagnostics:
    """
    Attributes:
        validation_error: available only if client.always_run_micro and
                client.compute_validation_error are both set to `True`
        stats_overhead_seconds: total time spent on stats computation, such as
                computing micro steps during the macro-only phase (if
                `client.always_run_micro` is `True`) and potentially expensive
                validation error computation, otherwise not present in
                production runs
    """
    start_timestep: int = 0
    cmp_error: Optional[Sequence[float]] = None
    cmp_uncertainty: Optional[Sequence[float]] = None
    validation_error: Optional[Sequence[float]] = None
    macro_steps: int = 0
    total_steps: int = 0
    stats_overhead_seconds: float = 0.0


@dataclass
class HistogramStatsConfig(DataclassMixin):
    """
    Arguments:
        hierarchy: (list, optional) If the data is a TensorCollection, its
                hierarchy must be specified in advance, such that the histogram
                can be properly initialized even when the dataset (training or
                validation) is empty. For now only flat TensorCollections are
                supported. In future, e.g. hierarchical structure could be
                added as a list of '/'-separated strings.
    """
    nbins: int
    range: Tuple[float, float]
    data: Union[Callable[[TrajectoryDataset], Any], str, Tuple[str, ...]]
    log: bool = False
    hierarchy: Optional[Dict[str, Optional[Dict]]] = None

    def compute(self, dataset: TrajectoryDataset):
        if isinstance(self.data, (str, tuple)):
            # Read as trajectories, because HDF5 dataset's as_states() does not
            # support slice access, and reading one state at a time would be
            # too expensive.
            if len(dataset.as_trajectories()) > 0:
                data = dataset.as_trajectories(self.data)
                data = [adaled.to_numpy(traj[:]) for traj in data]
                if isinstance(data[0], TensorCollection):
                    data = cmap(lambda *args: np.concatenate(args), *data)
                else:
                    data = np.concatenate(data)
            else:
                data = []
        else:
            data = self.data(dataset)
            data = adaled.to_numpy(data)

        if len(data) == 0:  # .shape may not be available if len is 0.
            histogram = np.zeros(self.nbins, dtype=np.int32)
            if self.hierarchy is not None:
                def rec(hier):
                    if hier is None:
                        return histogram  # zero
                    else:
                        return {key: rec(value) for key, value in hier.items()}
                histogram = rec(self.hierarchy)
            return histogram
        else:
            assert not isinstance(data, TensorCollection) \
                    or data.hierarchy() == self.hierarchy, \
                   f"if histogram data is a TensorCollection, hierarchy must " \
                   f"be specified and must match the data hierarchy: " \
                   f"data={data.hierarchy()}  self.hierarchy={self.hierarchy}"

        if self.log:
            data = cmap(np.log, data)
            range = np.log(self.range)
        else:
            range = self.range

        def func(x):
            data, bin_edges = np.histogram(x, self.nbins, range)
            return data.astype(np.int32)

        histogram = cmap(func, data)
        return histogram

    def get_bin_edges(self):
        if self.log:
            return np.exp(np.linspace(*np.log(list(self.range)), self.nbins + 1))
        else:
            return np.linspace(*self.range, self.nbins + 1)


def compute_dataset_stats(dataset: TrajectoryDataset,
                          histograms: Dict[str, HistogramStatsConfig]):
    """Return a dictionary of dataset sizes and histograms."""
    traj = len(dataset.as_trajectories())
    out = {
        'num_trajectories': np.int32(traj),
        'num_states': np.int32(len(dataset.as_states()) if traj else 0),
        'histograms': {k: h.compute(dataset) for k, h in histograms.items()},
    }
    return out


class AdaLEDDiagnostics:
    """Collection of diagnostics for visualization."""
    @staticmethod
    def load(path: str):
        data = io_.load(path)
        # Backward-compatibility, 2022-01-26, `data.dict` was stored, not `data`.
        data = TensorCollection(data)
        return AdaLEDDiagnostics(per_cycle_stats=DynamicArray(data))

    def __init__(self, *, per_cycle_stats=None):
        if per_cycle_stats is None:
            per_cycle_stats = DynamicArray()
        self.per_cycle_stats = per_cycle_stats
        self._lock: threading.Lock = threading.Lock()

        # For validation loss, keep track of the last non-nan value such that
        # we can print it on the screen nicely.
        self._last_non_nan_losses: Optional[TensorCollection] = None

        self.dataset_histograms: Dict[str, HistogramStatsConfig] = {}

    def state_dict(self) -> dict:
        with self._lock:
            data = copy.deepcopy(self.per_cycle_stats.data)
        return {
            'data': data,
            'last_non_nan_losses': self._last_non_nan_losses,
            # Do not add, they should be added during setup.
            # 'dataset_histograms': self.dataset_histograms
        }

    def load_state_dict(self, state: dict) -> None:
        self.per_cycle_stats.clear()
        self.per_cycle_stats.extend(state['data'])
        self._last_non_nan_losses = state['last_non_nan_losses']

    def add_dataset_histograms(self, histograms: Dict[str, HistogramStatsConfig]):
        assert len(self.dataset_histograms.keys() & histograms.keys()) == 0
        self.dataset_histograms.update(histograms)

    def save(self, path: str, verbose: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock:
            # We could in principle store `data.asdict()` to avoid adding
            # adaled as a dependency for postprocessing, but that then
            # complicates quick debugging.
            data = self.per_cycle_stats.data
            io_.save(data, path)
        if verbose:
            print(f"Saved AdaLED diagnostics to {path}.")

    def append_cycle(self, *args, **kwargs):
        with self._lock:
            self._append_cycle(*args, **kwargs)

    def _append_cycle(
            self,
            cycle: AdaLEDCycleDiagnostics,
            losses: TensorCollection,
            trainer_hyperparams: Dict[str, Any],
            datasets: DynamicTrajectoryDatasetCollection,
            **custom_stats):
        losses = losses.cpu()
        validation_error = cycle.validation_error
        if validation_error is None:
            validation_error = np.nan * cycle.cmp_error

        cycle_stats = TensorCollection({
            'end_wall_time': time.time(),
            'start_timestep': cycle.start_timestep,
            'cmp_error': cycle.cmp_error,
            'cmp_uncertainty': cycle.cmp_uncertainty,
            'validation_error': validation_error,
            'macro_steps': cycle.macro_steps,
            'stats_overhead_seconds': cycle.stats_overhead_seconds,
            'losses': losses,
            'trainer_hyperparams': trainer_hyperparams,
            'dataset': {
                'train': compute_dataset_stats(
                        datasets.train_dataset, self.dataset_histograms),
                'valid': compute_dataset_stats(
                        datasets.valid_dataset, self.dataset_histograms),
            },
            **custom_stats,
        }, default_numpy=True).cpu()
        self.per_cycle_stats.append(cycle_stats)
        self.log(losses)

    def log(self, losses: TensorCollection):
        """Print diagnostics information on the screen.

        Currently only prints the losses."""

        # Find last non-nan loss, useful for validation loss.
        if self._last_non_nan_losses is None:
            self._last_non_nan_losses = losses.clone()
        else:
            self._last_non_nan_losses = adaled.cmap(
                    lambda a, b: (a if np.isnan(b) else b),
                    self._last_non_nan_losses, losses)

        # Print transformer/macro stats only in their training loss is available.
        # fmt = lambda x: f'{x:12.6e}'
        fmt = lambda x: x
        losses = self._last_non_nan_losses

        tmp = losses.get('transformer_train', nan64)
        if isinstance(tmp, TensorCollection) or not np.isnan(to_numpy(tmp.sum())):
            print(f"Transformer losses: train={fmt(tmp)}\n"
                  f"                    valid={fmt(losses['transformer_valid'])}")
        tmp = losses.get('macro_train', nan64)
        if isinstance(tmp, TensorCollection) or not np.isnan(to_numpy(tmp.sum())):
            print(f"Macro losses: train={fmt(tmp)}\n"
                  f"              valid={fmt(losses['macro_valid'])}")


def load_diagnostics(output_dir: str) -> List[TensorCollection]:
    import glob
    pattern = os.path.join(output_dir, 'diagnostics-*.pt')
    paths = glob.glob(pattern)
    if not paths:
        # Try the legacy file name.
        try:
            return [TensorCollection(io_.load(
                    os.path.join(output_dir, 'diagnostics.pkl')))]
        except FileNotFoundError:
            pass
        raise FileNotFoundError(f"no diagnostics files found: {pattern}")

    return [TensorCollection(io_.load(path)) for path in paths]


def postprocess_diagnostics(
        diagnostics: Union[str, Sequence[TensorCollection]],
        avg_window: int,
        max_timesteps: int = 0):
    if isinstance(diagnostics, str):
        diagnostics = load_diagnostics(diagnostics)

    if len(diagnostics) > 1:
        raise NotImplementedError("postprocessing multi-rank diagnostics not "
                                  "implemented, process one rank a time instead")

    data: TensorCollection = diagnostics[0]
    if max_timesteps:
        end = np.searchsorted(data['start_timestep'], max_timesteps, side='right')
        data = data[:max(0, end - 1)]  # Cycle must end before max_timesteps.
        assert data['start_timestep'][-1] < max_timesteps, \
                (data['start_timestep'][-5:], max_timesteps)

    total_num_cycles = len(data)
    total_execution_time = data['end_wall_time'][-1] - data['end_wall_time'][0]
    total_fraction_macro = \
            data['macro_steps'][:-1].sum() \
            / (data['start_timestep'][-1] - data['start_timestep'][0])
    total_train_samples = data['total_train_samples']
    data = data[-avg_window:]
    data = data.cpu_numpy()
    try:
        cmp_error = data['cmp_error']
    except KeyError:
        cmp_error = data['cmp_mse']  # Legacy, 2021-11-20.

    losses = TensorCollection(losses=data['losses']).concat_flatten().map(np.nanmean)
    return {
        # Full simulation, no avg_window.
        'total_num_cycles': total_num_cycles,
        'total_execution_time': total_execution_time,
        'total_fraction_macro': total_fraction_macro,
        'total_train_samples_transformer': total_train_samples['transformer_train', -1],
        'total_train_samples_macro': total_train_samples['macro_train', -1],
        # Only the window part.
        'seconds_per_cycle': np.diff(data['end_wall_time']).mean(),
        'cmp_error': cmp_error.mean(),
        'cmp_uncertainty': data['cmp_uncertainty'].mean(),
        **losses.dict,
    }
