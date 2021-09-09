from adaled.backends import TensorCollection
from adaled.led.diagnostics import AdaLEDStage
from adaled.led.recording import RecorderConfig
from adaled.postprocessing.record import LazyRecordLoader
from adaled.solvers.base import MicroSolver
import adaled

import numpy as np

from typing import Sequence, Tuple
import os

__all__ = ['PostprocessRunner']

class PostprocessRunner:
    """Run the micro solver on existing recorded data in order to fill expected
    (ground truth) micro states during macro-only stages.

    Used as a replacement for AdaLEDClientConfig.always_run_micro == True,
    which would negatively affect the performance of the micro solver and thus
    affect the training process.
    """
    def __init__(
            self,
            micro: MicroSolver,
            config: RecorderConfig,
            paths: Sequence[str],
            sort_paths: bool = True):
        """
        Arguments:
            paths: list of record file paths
            sort_paths: (bool) sort paths lexicographically, defaults to True
        """
        if config.x_every != 1 or config.num_steps != config.start_every:
            raise ValueError(f"validation run requires x_every == 1 and "
                             f"num_steps == start_every got {config}")
        if sort_paths:
            paths = sorted(paths)

        self.micro = micro
        self.config = config
        self.paths = paths

        def _only_stages(multikey: Tuple[str, ...], d: 'h5py.Dataset'):
            return multikey == ('fields', 'metadata', 'stage')

        # TODO: Fix and use load_and_concat_records (currently something is broken there).
        loader = LazyRecordLoader(paths, load_immediately_regex=None)
        self.stages: np.ndarray = loader.small_fields['metadata', 'stage']
        print(f"Detected {len(self.stages)} steps.", flush=True)

        self._last_x_micro = None
        self._steps_to_skip = 0

    def run(self, verbose: bool = True):
        if not self.paths:
            if verbose:
                print("Nothing to run, paths list is empty.")
            return
        extensions = set(os.path.splitext(path)[1] for path in self.paths)
        if len(extensions) > 1:
            raise ValueError("all record files should have the same extension")
        extension = list(extensions)[0]
        if extension == '.h5':
            self._run_all_hdf5(verbose=verbose)
        else:
            raise NotImplementedError("currently only .h5 record files supported")

    def _run_all_hdf5(self, verbose: bool = True):
        import h5py  # Lazy import.
        steps_per_record = self.config.start_every
        for k, path in enumerate(self.paths):
            begin = k * steps_per_record
            end = begin + steps_per_record
            if verbose:
                print(f"Processing {path}  steps={begin}..{end - 1}", flush=True)
            with h5py.File(path, 'r+') as f:
                for j in range(steps_per_record):
                    self._next_step_hdf5(f, begin + j, j)

    def _next_step_hdf5(self, f: 'h5py.File', i: int, relative_i: int):
        from adaled.utils.io_hdf5 import load_hdf5_group

        # def _load_only_x_micro(multikey: Tuple[str, ...]):
        def _load_only_x_micro(d: 'h5py.Dataset'):
            multikey = tuple(d.name.split('/')[1:])
            if multikey[:4] == ('fields', 'simulations', 'x', 'micro'):
                return d[relative_i]
            else:
                return None  # Don't load.

        stages = self.stages

        # Example (W=warm-up, C=comparison, M=macro-only):
        #   W  W  C  C  C  C  M  M  M  (W  ...)
        #   0  1  2  3  4  5  6  7  8  (9  ...)
        #
        # Steps #0-5 are skipped (left as is).
        # For step #6, load step #5 with skip_steps == 6, advance and store.
        # Steps #7-9, we only advance and store.
        # ...

        if i > 0 and stages[i] == AdaLEDStage.MACRO:
            if stages[i - 1] == AdaLEDStage.COMPARISON:
                self.micro.update_state(self._last_x_micro,
                                        skip_steps=self._steps_to_skip)
                self._steps_to_skip = 0

            # TODO: F should be read from the record, it should NOT be a part
            #       of the micro solver!
            state = self.micro.advance(f['fields/simulations/F'][relative_i])
            self.get_state_and_update_record_hdf5(state, f, i, relative_i)
        else:
            self._steps_to_skip += 1

        if tuple(stages[i : i + 2]) == (AdaLEDStage.COMPARISON, AdaLEDStage.MACRO):
            self._last_x_micro = \
                    load_hdf5_group(f, _load_only_x_micro) \
                    ['fields', 'simulations', 'x', 'micro']

    # virtual
    def get_state_and_update_record_hdf5(
            self, state: TensorCollection, f: 'h5py.File', step: int, relative_step: int):
        """Get the state from the micro solver and update the record for the
        corresponding step."""
        from adaled.utils.io_hdf5 import update_hdf5_group
        state = adaled.to_numpy(state)
        group = f['fields']['simulations']['x']['micro']
        update_hdf5_group(group, state, slice=relative_step)
