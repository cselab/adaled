from adaled import TensorCollection, DynamicArray, get_backend
from adaled.led import AdaLEDStage as Stage
from adaled.backends.generic import cmap, extended_nanlike
from adaled.nn.loss import weighted_mse_losses
from adaled.postprocessing.record import LazyRecordLoader, get_cycle_slices, \
        load_and_concat_records
from adaled.utils.io_hdf5 import load_hdf5
import adaled
import adaled.postprocessing.run
from .config import CombinedConfigBase
from .loss import ReconstructionLoss
from .micro import AutoencoderReconstructionHelper, CUPTensorCollection, \
        MicroStateHelper, MicroStateType

import numpy as np
import torch
import tqdm

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Type
import argparse
import glob
import os
import re
import warnings

def _relative_mse(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    assert value.ndim == 3 and value.shape[0] == 2 and value.shape == target.shape, \
            (value.shape, target.shape)
    value = torch.from_numpy(value)[np.newaxis]
    target = torch.from_numpy(target)[np.newaxis]
    out = weighted_mse_losses(value, target, weight=None, relative=True, eps=0.0)
    return out.numpy()[0]


def load_cup_forces(
        path: str = 'cubismup/forceValues_0.dat',
        min_t: float = -np.inf,
        max_t: float = +np.inf) -> TensorCollection:
    """Load the forceValues file and return the columns and merged columns as a
    TensorCollection."""
    data = adaled.load_csv('cubismup/forceValues_0.dat', delimiter=None)
    t = data['time']

    lo = np.searchsorted(t, min_t)
    hi = np.searchsorted(t, max_t)
    t = t[lo:hi]
    data = data[lo:hi]
    columns = ['FxPres', 'FyPres', 'FxVisc', 'FyVisc']
    forces = np.stack([data[col] for col in columns], axis=-1)

    return TensorCollection({
        't': t,
        'FPres': forces[:, 0:2],
        'FVisc': forces[:, 2:4],
        'FPresVisc': forces,
    })


def find_records() -> Dict[int, List[str]]:
    """Search for record files and organize them per sim_id.

    Returns a dictionary:
        {sim_id: sorted list of record paths}.
    """
    paths = sorted(glob.glob('record-0*-0*.h5'))
    sim_paths = defaultdict(list)
    pattern = re.compile(r'^record-(\d+)-\d+.h5$')
    for path in sorted(paths):
        match = pattern.match(path)
        sim_id = int(match.group(1))
        sim_paths[sim_id].append(path)

    return sim_paths


def parse_execution_time_from_stderr(dir: str = '.') -> np.ndarray:
    """Parse STEP logs from stderr files and return an array a of shape Nx2,
    where a[i] = (cup execution time, total execution time).

    Missing steps are filled with NaNs.

    Args:
        dir (str): run folder
    """
    pattern = re.compile(r'STEP #(\d+):\W*([0-9.]+)s\W*([0-9.]+)s')

    step_ids = []
    t1 = []
    t2 = []
    for path in glob.glob(os.path.join(dir, 'stderr-*.txt')):
        with open(path) as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    step_ids.append(int(m.group(1)))
                    t1.append(float(m.group(2)))
                    t2.append(float(m.group(3)))
    step_ids = np.array(step_ids)
    rhs = np.stack([t1, t2], axis=-1)

    out = np.full((step_ids.max() + 1, 2), np.nan)
    out[step_ids, :] = rhs
    return out


def compute_cumulative_stats(post: TensorCollection) -> dict:
    """Compute cumulative stats, useful for e.g. hyperparameter comparison with
    hiplot."""
    out = {}
    for key, value in post.concat_flatten().items():
        if 'micro_macro' in key or 'errors' in key:
            out[f'postprocess.mean.{key}'] = np.nanmean(value)
    return out


class Postprocess:
    """Helper class for postprocessing.

    Computes:
        - pressure and viscous forces
        - macro <-> micro error
        - micro <-> micro shifted by one frame error, as an error baseline

    Works only for 2D simulations.
    """
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument_group('postprocessing').add_argument
        add('--stride', type=int, default=1, help="use only every stride-th record sample")
        add('--slice', type=int, nargs=2, default=(0, -1), help="section to use")
        add('--ref-frame', type=int,
            help="Compute loss between current and --ref-frame x_micro frame. "
                 "Used as a second baseline, to estimate largest difference "
                 "between two frames.")

        what_options = ['errors', 'x_forces', 'manual_forces', 'losses']
        add('--what', type=str, nargs='*',
            default=what_options, choices=what_options,
            help="what to postprocess, available options: " + ", ".join(what_options))

        add('--variant', choices=('online', 'offline'), default='online',
            help="Online: only concatenate data that was already computed with  "
                 "always_run_micro=1 (--stride and --slice support NOT implemented), "
                 "Offline: legacy expensive record-based postprocessing")

    def __init__(self, config: CombinedConfigBase):
        self.config = config
        self.ae_helper = AutoencoderReconstructionHelper(config.micro)
        self.micro_helper = MicroStateHelper(config.micro)
        self.mesh_x, self.mesh_y = config.micro.compute_cell_centers()
        self.mesh_dx = (self.mesh_x[1:] + self.mesh_x[:-1]) / 2
        self.mesh_dy = (self.mesh_y[1:] + self.mesh_y[:-1]) / 2
        self.mr = config.micro.make_multiresolution()

        self.segments_p = self.get_forces_integration_segments(lift_h=0.0)
        self.segments_v = self.get_forces_integration_segments(lift_h=2.0)

    def get_forces_integration_segments(self, lift_h: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return forces integration segments as a tuple of three arrays:
            - Nx2 array of coordinates,
            - Nx2 array of normals,
            - N array of segment lengths.
        """
        raise NotImplementedError(self)

    def find_and_postprocess(self, args):
        """Search for all record files and postprocess them."""
        mr = self.config.micro.make_multiresolution()

        for sim_id, paths in find_records().items():
            print(f"Postprocessing simulation #{sim_id}.")
            if args.variant == 'online':
                # TODO: Rename runtime to online.
                data = self.postprocess_one_simulation__online(paths)
                path = f'postprocessed-runtime-{sim_id:03d}.pt'
            else:
                data = self.postprocess_one_simulation__offline(
                        paths, log_prefix=f"sim_id={sim_id:03d}",
                        what=args.what, stride=args.stride,
                        start=args.slice[0], end=args.slice[1],
                        ref_frame=args.ref_frame)
                path = f'postprocessed-{sim_id:03d}.pt'

            print("New results to be saved:")
            data.describe()
            try:
                old: TensorCollection = adaled.load(path)
                print("Merging with old data (new has priority):")
                old.describe()
                for multikey, value in old.allitems():
                    curr = data
                    for key in multikey[:-1]:
                        if key not in curr:
                            curr[key] = {}
                        curr = curr[key]
                    if multikey[-1] not in curr:
                        curr[multikey[-1]] = value
            except FileNotFoundError:
                old = None

            adaled.save(data, path)

    def postprocess_one_simulation__online(self, paths: Sequence[str]) -> TensorCollection:
        """Postprocess stats collected with always_run_micro=1."""
        def filter(k: Tuple[str, ...], d: 'h5py.Dataset'):
            if k == ('fields', 'simulations', 'F'):
                return d[:]
            if 'metadata' in k \
                    or 'x_micro_qoi' in k \
                    or ('z' in k and 'macro' in k):
                return d[:]
            if 'validation' in k:
                if 'cmp_error' in k or 'vorticity' in k:
                    return d[:]
            return None

        record: TensorCollection = load_and_concat_records(paths, filter)['fields']
        record.describe()
        sim = record['simulations']
        out = {
            'metadata': record['metadata'],
            'micro_recorded_qoi': sim['x_micro_qoi'],
            'micro_state_qoi': \
                    self.config.micro.record_qoi_to_state_qoi(sim['x_micro_qoi']),
            'F': sim['F'],
        }
        if 'validation' in sim:
            out['cmp_error'] = {
                'v': sim['validation', 'cmp_error', 'v'],
                **sim['validation'].named_map(
                    lambda multikey, value: value if 'vorticity' in multikey else None)
            }
        out['transition_qoi_errors'] = self.compute_transition_qoi_error(
                record['metadata', 'stage'], out['micro_state_qoi'])

        qoi_z_size = self.config.extra_latent_size()
        if qoi_z_size > 0:
            transformer: adaled.CompoundAutoencoder = adaled.load('transformer-latest.pt')
            z_macro = torch.from_numpy(sim['z', 'macro'])
            assert z_macro.ndim == 3 and z_macro.shape[1] == 1, \
                   z_macro.shape  # (duration, batch == 1, latent space)
            # TODO: Do not hardcode cyl here. Add a separate 'qoi' field.
            macro_qoi = {}
            for key in ['cyl_forces']:
                assert z_macro.shape[1] == 1, z_macro.shape
                macro_qoi[key] = transformer.partial_inverse_transform(
                        z_macro[:, 0], key)[:, np.newaxis]
            out['macro_qoi'] = macro_qoi

        return TensorCollection(out).cpu_numpy()

    def compute_transition_qoi_error(
            self,
            stages: np.ndarray,
            qoi: TensorCollection,
            degree: int = 2,
            history: int = 5) -> TensorCollection:
        """Compute the error of QoI after the transition.

        Since we do not have ground truth data for warm-up micro time steps (as
        a continuation of previous macro phase), we fit a polynomial on micro
        QoI during the macro-only phase (assuming always_run_micro=1) and
        extrapolate that.

        To verify how reliable the fit itself is, we also estimate the micro
        QoI at the last macro-only step using the same procedure.
        """
        def extrapolate(data: np.ndarray):
            assert 2 <= data.ndim <= 3 and data.shape[1] == 1, data.shape
            is_vector = data.ndim == 3
            if is_vector:
                data = data[:, 0, :]
            x = np.arange(len(data))
            poly, var = np.polyfit(x, data, degree, cov=True)
            out = np.nan * data[-1]
            for i in range(len(out)):
                out[i] = np.poly1d(poly[:, i])(len(data))

            out = out[np.newaxis, :] if is_vector else out  # Add batch back.
            return out

        def l2(a, b):
            return ((a - b) ** 2).sum(axis=-1) ** 0.5

        # Remove last, possibly incomplete, cycle.
        cycles = get_cycle_slices(stages)[:-1]
        is_accepted = stages[cycles[:, 1] - 1] == Stage.MACRO

        out = adaled.DynamicArray()
        for i, (begin, end) in enumerate(cycles):
            if not is_accepted[i]:
                continue
            extra0 = cmap(extrapolate, qoi[end - history : end])
            extra1 = cmap(extrapolate, qoi[end - history - 1 : end - 1])
            out.append(TensorCollection({
                'end': end,
                'extrapolated_first_warmup': extra0,
                'extrapolated_last_macro_only': extra1,
                'actual_first_warmup': qoi[end],
                'actual_last_macro_only': qoi[end - 1],
                'l2_first_warmup': l2(extra0, qoi[end]),
                'l2_last_macro_only': l2(extra1, qoi[end - 1]),
            }))
        out = out.data
        if len(out) > 0:
            for multikey, x in out['l2_first_warmup'].allitems():
                print("l2_first_warmup   ", multikey, x.mean(), x.std())
            for multikey, x in out['l2_last_macro_only'].allitems():
                print("l2_last_macro_only", multikey, x.mean(), x.std())
        else:
            print("No accepted cycles.")
        return out


    def postprocess_one_simulation__offline(
            self,
            paths: Sequence[str],
            what: List[str],
            stride: Optional[int] = None,
            start: int = 0,
            end: int = -1,
            ref_frame: Optional[int] = None,
            log_prefix: str = ""):
        """Postprocess record files of one simulation."""
        pattern = r'/?fields/simulations/x/(micro|macro)/\w*forces'
        loader = LazyRecordLoader(paths, load_immediately_regex=pattern)
        record = loader.small_fields
        print("Record structure:")
        record.describe()

        if ref_frame:
            print(f"Computing loss with respect to reference frame #{ref_frame}.")
            ref_x_micro = loader.get_frame_x(ref_frame)['x', 'micro']
        else:
            print(f"--ref-frame not specified, not computing reference loss.")

        mr = self.config.micro.make_multiresolution()
        loss_macro = ReconstructionLoss(self.config, mr)
        loss_micro = ReconstructionLoss(self.config, mr, is_macro=False)

        metadata_every = self.config.led.recorder.every
        x_every = self.config.led.recorder.x_every
        if stride is None:
            stride = x_every
        if end < 0:
            end += len(record) + 1
        if start % x_every != 0 or end % x_every != 0 or stride % x_every != 0:
            raise ValueError(f"start={start}, end={end} and stride={stride} "
                             f"must be a multiple of x_every={x_every}")

        metadata_every = self.config.led.recorder.every
        x_every = self.config.led.recorder.x_every
        length = (end - start) * metadata_every // x_every

        out = {}
        if 'errors' in what:
            out['errors'] = {}
            out['errors']['rel_v_no_eps'] = rvne = {
                'consecutive': np.full(length, np.nan),
                'micro_macro': np.full(length, np.nan),
            }
            if self.config.micro.record_full_resolution:
                rvne['full_micro_scaling'] = np.full(length, np.nan)
                rvne['full_micro_consecutive'] = np.full(length, np.nan)
                rvne['full_micro_downscaled_macro'] = np.full(length, np.nan)
        if 'x_forces' in what:
            def _get_forces(x):  # A bit hacky.
                return {k: v for k, v in x.items() if 'forces' in k}
            try:
                out['x_forces'] = {
                    'micro': _get_forces(record['simulations', 'x', 'micro']),
                    'macro': _get_forces(record['simulations', 'x', 'macro']),
                }
            except KeyError:
                pass
        if 'manual_forces' in what:
            out['manual_forces'] = manual_forces = {
                'micro': np.empty((length, 4)),
                'macro': np.empty((length, 4)),
            }
        if 'losses' in what:
            backend = get_backend(np.arange(5.))
            out['losses'] = losses = {
                'consecutive': extended_nanlike(loss_micro.get_zero_loss(backend), length),
                'micro_macro': extended_nanlike(loss_macro.get_zero_loss(backend), length),
            }
            if ref_frame is not None:
                out['losses']['ref_frame'] = out['losses']['consecutive'] * np.nan  # Copy.

        def filter_load_full(multikey: Tuple[str, ...]):
            if multikey[:3] == ('fields', 'simulations', 'x_micro_full_resolution'):
                return True  # Yes, load.
            else:
                return None  # Use default criterion.

        last_x_micro = None
        last_v_macro = None
        last_v_micro = None
        last_v_micro_full = v_micro_full = None
        for ii in tqdm.trange(start, end, stride, desc=log_prefix):
            if ii == 0 and not (set(what) & set(['errors', 'manual_forces', 'losses'])):
                print("Nothing to process on per-frame basis, skipping.")
                break
            x = loader.get_frame_x(ii, filter_func=filter_load_full)
            i = ii // stride

            x_micro = x['x', 'micro']
            x_macro = x['x', 'macro']
            v_micro, p_micro = self.rebuild_frame(x_micro, MicroStateType.GET_STATE)
            v_macro, p_macro = self.rebuild_frame(x_macro, MicroStateType.DECODER)
            if self.config.micro.record_full_resolution:
                v_micro_full = self.micro_helper.layer_to_velocity(
                        x['x_micro_full_resolution'])[0]

            if stride > 1 and ('errors' in what or 'losses' in what):
                warnings.warn(f"Computing error/loss not between consecutive, "
                              f"but between strided frames (stride={stride})!")
            if 'errors' in what and last_v_micro is not None:
                rvne['consecutive'][i] = _relative_mse(last_v_micro, v_micro)
                rvne['micro_macro'][i] = _relative_mse(v_macro, v_micro)
                if self.config.micro.record_full_resolution:
                    rvne['full_micro_scaling'][i] = \
                            _relative_mse(v_micro, v_micro_full)
                    rvne['full_micro_consecutive'][i] = \
                            _relative_mse(last_v_micro_full, v_micro_full)
                    rvne['full_micro_downscaled_macro'][i] = \
                            _relative_mse(v_macro, v_micro_full)
            if 'manual_forces' in what:
                manual_forces['micro'][i, 0:2] = self.compute_pressure_force(p_micro)
                manual_forces['micro'][i, 2:4] = self.compute_viscous_force(v_micro)
                manual_forces['macro'][i, 0:2] = self.compute_pressure_force(p_macro)
                manual_forces['macro'][i, 2:4] = self.compute_viscous_force(v_macro)
            if 'losses' in what and last_v_micro is not None:
                losses['consecutive'][i] = loss_micro(last_x_micro, x_micro)
                losses['micro_macro'][i] = loss_macro(x_macro, x_micro)
                if ref_frame is not None:
                    losses['ref_frame'][i] = loss_micro(ref_x_micro, x_micro)

            last_v_macro = v_macro
            last_v_micro = v_micro
            last_v_micro_full = v_micro_full
            last_x_micro = x_micro

        return TensorCollection({
            'metadata': record['metadata', start : end : stride // metadata_every],
            'postprocess': out,
        })

    def rebuild_frame(self,
                      state: TensorCollection,
                      state_type: MicroStateType) -> np.ndarray:
        assert len(state) == 1, state.shape  # Batch size 1.
        layers = state['layers']
        layers = [layers[key] for key in sorted(layers.keys())]
        if state_type == MicroStateType.DECODER:
            v_layers = [self.ae_helper.layer_to_velocity(i, layer)
                        for i, layer in enumerate(layers)]
            p_layers = [self.ae_helper.layer_to_pressure(layer) for layer in layers]
        else:
            v_layers = [self.micro_helper.layer_to_velocity(layer) for layer in layers]
            p_layers = [self.micro_helper.layer_to_pressure(layer) for layer in layers]

        v = self.mr.rebuild_unsafe(v_layers)[0]
        p = self.mr.rebuild_unsafe(p_layers)[0] if self.config.micro.pressure_in_state else None
        return v, p

        out = [self.compute_forces(v[i], p[i]) for i in range(len(v))]
        return np.stack(out)

    def compute_pressure_force(self, p: Optional[np.ndarray]) -> Tuple[float, float]:
        """Integrate the pressure force on a (y, x)-shaped pressure field."""
        if p is None:
            return np.nan
        from scipy.interpolate import RegularGridInterpolator
        assert p.ndim == 2, p.shape
        x, n, dl = self.segments_p
        pressure = RegularGridInterpolator((self.mesh_x, self.mesh_y), p.T)(x)
        F_pressure = (-pressure * n.T * dl).T.sum(axis=0)
        return F_pressure

    def compute_viscous_force(self, v: np.ndarray) -> float:
        """Integrate the viscous force on a (2, y, x)-shaped velocity field."""
        from scipy.interpolate import RegularGridInterpolator
        assert v.ndim == 3 and v.shape[0] == 2, v.shape
        x, n, dl = self.segments_v
        h = self.config.micro.compute_h()
        dvx_dx = (v[0, :, 1:] - v[0, :, :-1]) / h
        dvy_dx = (v[1, :, 1:] - v[1, :, :-1]) / h
        dvx_dy = (v[0, 1:, :] - v[0, :-1, :]) / h
        dvy_dy = (v[1, 1:, :] - v[1, :-1, :]) / h
        dvx_dx = RegularGridInterpolator((self.mesh_dx, self.mesh_y), dvx_dx.T)(x)
        dvy_dx = RegularGridInterpolator((self.mesh_dx, self.mesh_y), dvy_dx.T)(x)
        dvx_dy = RegularGridInterpolator((self.mesh_x, self.mesh_dy), dvx_dy.T)(x)
        dvy_dy = RegularGridInterpolator((self.mesh_x, self.mesh_dy), dvy_dy.T)(x)

        # Not sure if this is the correct order, but it doesn't matter due to nv+nvT.
        nabla_v = np.array([[dvx_dx, dvy_dx], [dvx_dy, dvy_dy]]).T
        nabla_v_T = np.moveaxis(nabla_v, 1, 2)
        samples = np.einsum('ijk,ik->ij', nabla_v + nabla_v_T, n)
        F_viscous = self.config.micro.nu * (samples.T * dl).T.sum(axis=0)
        return F_viscous


class PostprocessRunnerEx(adaled.postprocessing.run.PostprocessRunner):
    """Extend existing PostprocessRunner such that x_micro_full_resolution is
    also stored when micro.record_full_resolution=1."""
    def get_state_and_update_record_hdf5(
            self, state: CUPTensorCollection,
            f: 'h5py.File', step: int, relative_step: int, **kwargs):
        super().get_state_and_update_record_hdf5(state, f, step, relative_step, **kwargs)
        micro: CUPSolverBase = self.micro
        if micro.config.record_full_resolution:
            from adaled.utils.io_hdf5 import update_hdf5_group
            group = f['fields']['simulations']['x_micro_full_resolution']
            update_hdf5_group(group, state.full_resolution_state, slice=relative_step)


def main(cls: Type[Postprocess], argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    cls.add_arguments(parser)
    args = parser.parse_args(argv)

    adaled.init_torch()
    config: CombinedConfigBase = adaled.load('config.pt')
    postprocess = cls(config)
    postprocess.find_and_postprocess(args)
