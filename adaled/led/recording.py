from adaled.backends import TensorCollection
from adaled.led.client import AdaLEDStep
from adaled.nn.networks import RichHiddenState
from adaled.utils.buffer import DynamicArray
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.misc import to_numpy
from adaled.utils.transformations import \
        _CompoundTransformation, Transformation, parse_transformation
import adaled

import numpy as np
import torch

from typing import Any, Dict, Optional
import dataclasses
import math
import os
import time

@dataclass
class RecorderConfig(DataclassMixin):
    """Recorder configuration.

    Attributes:
        every: frequency of storing metadata (time step index, stage, execution
               time) and small fields (F, uncertainty) (*)
        x_every: frequency of storing `x` data; if set to 0 (default), the
                 value of `every` is taken (*)
        z_every: frequency of storing `z` data; if set to 0 (default), the
                 value of `every` is taken (*)

    Transformation arguments, define which data transformations to apply on the
    data before storing:

    Attributes:
        pretransform: applied to all, first
        posttransform: applied to all, last
        transform_x_micro: applied to x_micro only, before transform_x
        transform_x_macro: applied to x_macro only, before transform_x
        transform_x: applied to x_micro and x_macro after transform_x_micro and
                transform_x_macro, respectively
        transform_z: applied to z_micro and z_macro
        transform_F: applied to F
        transform_raw_macro_output: applied to the raw output of the macro
                solver, used to analyze the macro ensemble

    (*) Values `x_every` and `z_every` must be multiples of `every`. Value
    `num_steps` must be a multiple of `every / x_every` and `every / z_every`.
    """
    start_every: int = 0
    num_steps: int = 0

    # Note: for now no further fine-tuning is implemented. The `x` space is
    # potentially very large and it makes sense to save it a lower frequency
    # than the rest. Likewise, with CRNN the `z` state is large as well, so it
    # makes sense to control it as well. For the rest the benefit doesn't seem
    # to be great.
    every: int = 1
    x_every: int = 0
    z_every: int = 0

    # Path format with the following variables:
    #   start_timestep: the time step at which the recording starts
    #   block_index: which recording it is
    path_fmt: Optional[str] = None

    # Path of the symlink to the latest trajectory dump file.
    latest_symlink_fmt: Optional[str] = '<auto>'

    verbose: bool = True

    # Transformations to apply to the data before storing.
    pretransform: Transformation = 'identity'
    posttransform: Transformation = 'identity'
    transform_x_micro: Transformation = 'identity'
    transform_x_macro: Transformation = 'identity'
    transform_x: Transformation = 'identity'
    transform_z: Transformation = 'identity'
    transform_F: Transformation = 'identity'
    transform_raw_macro_output: Transformation = 'identity'

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'x_every'):  # Backward compatibility, 2022-02-04.
            self.x_every = self.every
            self.z_every = self.every

    def __post_init__(self):
        if self.x_every == 0:
            self.x_every = self.every
        if self.z_every == 0:
            self.z_every = self.every

    def validate(self, *args, **kwargs):
        super().validate(*args, **kwargs)
        if self.start_every > 0 and self.start_every < self.num_steps * self.every:
            raise ValueError(f"start_every={self.start_every} must be 0 or "
                             f"larger than or equal to num_steps*every="
                             f"{self.num_steps}*{self.every}")
        for attr in ['x_every', 'z_every']:
            if getattr(self, attr) % self.every != 0:
                raise ValueError(f"{attr}={getattr(self, attr)} must be a "
                                 f"multiple of every={self.every}")
        gcd = math.gcd(self.x_every, self.z_every)
        for attr in ['num_steps', 'start_every']:
            if getattr(self, attr) % gcd != 0:
                raise ValueError(f"{attr} must be divisible by every, x_every "
                                 f"and z_every\nself={self}")


class TrajectoryRecorder:
    # Version of the record information. In case it turns out useful for
    # postprocessing and plotting scripts.
    RECORD_VERSION = 6

    def __init__(self,
                 config: RecorderConfig,
                 sim_id: int = 0,
                 extra_sim_records: Dict[str, DynamicArray] = {}):
        """
        Arguments:
            sim_id: client ID
        """
        self.config = config
        self.record_metadata = DynamicArray()
        self.record_F = DynamicArray()
        self.record_uncertainty = DynamicArray()
        self.record_x = DynamicArray()
        self.record_z = DynamicArray()
        self._ref = {}
        self.sim_id = sim_id
        self.sim_records = {
            # record_metadata not included, it does not contain the batch
            # dimension, see prepare_data_for_saving
            'F': self.record_F,
            'uncertainty': self.record_uncertainty,
            'x': self.record_x,
            'z': self.record_z,
            **extra_sim_records,
        }

        _parse = parse_transformation
        pretransform = _parse(config.pretransform)
        posttransform = _parse(config.posttransform)
        transform_x = _parse(config.transform_x)

        def _wrap(*t):
            t = [_parse(_) for _ in t]
            return _CompoundTransformation(pretransform, *t, posttransform)

        self._transform_x_micro = _wrap(config.transform_x_micro, transform_x)
        self._transform_x_macro = _wrap(config.transform_x_macro, transform_x)
        self._transform_z = _wrap(config.transform_z)
        self._transform_F = _wrap(config.transform_F)
        self._transform_raw_macro_output = _wrap(config.transform_raw_macro_output)
        self._prev_time = np.nan

    def flush(self, i: int):
        config = self.config
        block_index = i // config.start_every if config.start_every else 0
        path = self._format(config.path_fmt, block_index)
        data = self.prepare_data_for_saving()
        adaled.save(data, path, verbose=config.verbose)
        if config.verbose:
            len_ = len(self.record_metadata)
            print(f"Stored {len_} frames representing {len_ * config.every} "
                  f"timesteps to {path}.")
        sympath = config.latest_symlink_fmt
        if sympath:
            if sympath != '<auto>':
                sympath = self._format(sympath, block_index)
            adaled.make_symlink(path, sympath, hint_fmt=config.path_fmt,
                                verbose=config.verbose)
        self.record_metadata.clear()
        for record in self.sim_records.values():
            record.clear()

    def state_dict(self) -> dict:
        return {
            'ref': self._ref,
            'sim_id': self.sim_id,
            'record_metadata': self.record_metadata.data,
            'sim_records': {key: record.data for key, record in self.sim_records.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.record_metadata.clear()
        self.record_metadata.extend(state['record_metadata'])
        for key, record in self.sim_records.items():
            record.clear()
            record.extend(state['sim_records'][key])
        self._ref = state['ref']
        self.sim_id = state['sim_id']

    def prepare_data_for_saving(self):
        """Return the object to be stored on disk."""
        data = TensorCollection({
            'version': self.RECORD_VERSION,
            # Fixme: Are there any more appropriate names that wouldn't
            # conflict with the rest of the concepts?
            'fields': { # Anything with time as the first dimension. Different
                        # fields may be recorded at different frequencies!
                # Anything that is constant along the batch.
                'metadata': self.record_metadata.data,
                # Anything with batch as second dimension.
                'simulations': {
                    key: record.data for key, record in self.sim_records.items()
                },
            },
        })
        return data

    def _format(self, fmt: str, block_index: int):
        """Format record path."""
        return fmt.format(
                start_timestep=block_index * self.config.start_every,
                block_index=block_index, rank=self.sim_id, sim_id=self.sim_id)

    def add_step(self, i: int, step: AdaLEDStep):
        config = self.config
        relative_i = i % config.start_every if config.start_every > 0 else i

        raw_output = self._read_raw_output(step)
        if self._check_reference_values(i, step, raw_output) \
                and relative_i % config.every == 0 \
                and (config.num_steps == 0 or \
                     relative_i < config.num_steps * config.every):
            self.record_step(i, relative_i, step, raw_output)
            if relative_i == (config.num_steps - 1) * config.every:
                self.flush(i)

    def _read_raw_output(self, step: AdaLEDStep):
        """Attempt to read raw network output from its hidden state."""
        h = step.hidden
        if h is None:
            return None  # Raw output not known yet.
        if isinstance(h, RichHiddenState):
            return h.raw_output
        else:
            return np.zeros(())  # Raw output now known to be unavailable.

    def _check_reference_values(
            self, i: int, step: AdaLEDStep, raw_output: Optional[Any]) -> bool:
        NUM_REF_VALUES = 4  # x, z, raw, uncertainty
        ref = self._ref
        # FIXME: It turns out that x, z and raw output are available after the
        # first time step, so this whole logic of checking what is available is
        # not needed and should be removed. In fact, skipping time steps breaks
        # the plotting script, because it expects a simple mapping from time
        # steps to record files. For now we simply hard-code uncertainty to a
        # single number.
        if len(ref) == NUM_REF_VALUES:
            return True  # Reference values already known.
        if step.x is not None:
            ref.setdefault('x', step.x)  # x_micro
        if step.z is not None:
            ref.setdefault('z', step.z)
            ref.setdefault('uncertainty', np.full(len(step.z), np.nan))
        # if step.uncertainty is not None:
        #     ref.setdefault('uncertainty', step.uncertainty)
        if raw_output is not None:
            ref.setdefault('raw', raw_output)
        if len(ref) < NUM_REF_VALUES:
            return False  # Not all reference values known yet.

        config = self.config
        num_simulations = len(ref['x'])
        if raw_output is not None:
            assert raw_output.shape[0] == num_simulations
        assert len(step.F) == num_simulations, \
               "F must be given as an array over simulations"
        with torch.no_grad():
            trans_ref_x_macro = self._transform_x_macro(
                    step.transformer.inverse_transform(
                        step.transformer.transform(ref['x'])))
        trans_ref_x = self._transform_x_micro(ref['x'])
        trans_ref_z = self._transform_z(ref['z'])
        trans_ref_raw = self._transform_raw_macro_output(ref['raw'])
        # Add one dummy element and remove it immediately to initialize
        # the buffers, i.e. specify their types and backends.
        self.record_metadata.append(TensorCollection({
            'timestep': np.int32(i),
            'stage': np.int8(step.stage),
            'execution_time': np.float32(np.nan),
        }))
        self.record_F.append(to_numpy(self._transform_F(step.F)))
        self.record_uncertainty.append(to_numpy(ref['uncertainty']))
        self.record_x.append(TensorCollection({
            'micro': trans_ref_x,
            'macro': trans_ref_x_macro,  # Reconstructed (z->x).
        }).cpu_numpy())
        self.record_z.append(TensorCollection({
            'micro': trans_ref_z,  # Transformed x->z.
            'macro': trans_ref_z,
            'raw_macro': trans_ref_raw,
        }).cpu_numpy())
        self.record_metadata.clear()
        for record in self.sim_records.values():
            record.clear()
        return True

    def record_step(
            self,
            i: int,
            relative_i: int,
            step: AdaLEDStep,
            raw_output: Optional[Any]) -> None:
        config = self.config
        record_x = i % self.config.x_every == 0
        record_z = i % self.config.z_every == 0

        with torch.no_grad():
            if record_z and step.x is not None:
                trans_z_micro = self._transform_z(
                        step.transformer.transform(step.x))
            else:
                trans_z_micro = np.nan

            # x_macro will be the reconstruction x in the micro-only stages.
            if record_x or record_z:
                z_macro = step.z if step.z is not None \
                          else step.transformer(step.x)
            if record_x:
                x_macro = step.transformer.inverse_transform(z_macro)
                trans_x_macro = self._transform_x_macro(x_macro)

        current_time = time.time()
        self.record_metadata.append(TensorCollection({
            'timestep': np.int32(i),
            'stage': np.int8(step.stage),
            'execution_time': np.float32(current_time - self._prev_time),
        }))
        self.record_F.append(to_numpy(self._transform_F(step.F)))
        self.record_uncertainty.append(
                np.nan if step.uncertainty is None else to_numpy(step.uncertainty))
        if record_x:
            self.record_x.append(TensorCollection({
                'micro': np.nan if step.x is None \
                        else self._transform_x_micro(step.x),
                'macro': trans_x_macro,
            }).cpu_numpy())
        if record_z:
            self.record_z.append(TensorCollection({
                'micro': trans_z_micro,
                'macro': self._transform_z(z_macro),
                'raw_macro': np.nan if raw_output is None \
                             else self._transform_raw_macro_output(raw_output),
            }).cpu_numpy())

        self._prev_time = current_time
