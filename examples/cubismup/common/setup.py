from .config import AutoencoderConfig, DatasetConfig, MacroConfig, \
        MicroConfigBase, CombinedConfigBase
from .loss import ConcatVorticity2DLayer, ConcatVorticity3DLayer, \
        FullResolutionComparisonError, ReconstructionLoss
from .micro import CUPSolverBase, CUPTensorCollection
from .utils_2d import compute_vorticity_2d_no_boundary
from .utils_3d import curl
import adaled
from adaled import AdaLEDStage, AdaLEDStep, TensorCollection, Topology
# TODO: rename ServerClientMain to something else, there's no main there.
from adaled.led.main import ServerClientMain
from adaled.transformers.multiresolution import Multiresolution
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass

import numpy as np
import torch

from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import argparse
import os
import platform
import shlex
import sys
import time

if TYPE_CHECKING:
    from mpi4py import MPI

MOVIE_SCRIPT_TEMPLATE = '''\
#!/bin/bash

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 {path} "$@"
'''

PLOTTING_SCRIPT_TEMPLATE = '''\
#!/bin/bash

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 {path} "$@"
'''

POSTPROCESS_SCRIPT_TEMPLATE = '''\
#!/bin/bash

python3 {path} "$@"
'''

class ConvDecoder(adaled.ConvMLPDecoder):
    def __init__(self,
                 config: CombinedConfigBase,
                 encoder: adaled.ConvMLPEncoder):
        kwargs = {}
        if config.micro.predict_potential_function or config.micro.predict_pressure:
            # This was originally enabled only when predicting the potential
            # function, but using it for pressure as well. Probably it was
            # added because in case of a potential, incorrectly tuned scaling
            # and non-linearity might cause issues.
            kwargs['activation_output'] = 'none'

        self.predict_potential_function = config.micro.predict_potential_function
        super().__init__(encoder.make_decoder_config(
                output_channels=config.get_num_ae_output_channels(), **kwargs))

    def make_padding(self, layer: int):
        """Add +1 padding on the last layer if predicting stream fn, such that
        the reconstructed velocity has the correct shape. This performs much
        better than extrapolating the velocity at the boundary."""
        # In case both predict_potential_function and predict_pressure are
        # True, we will have dummy padding for the pressure.
        padding, padding_mode, padding_layers, conv_kwargs = super().make_padding(layer)
        num_layers = len(self.config.conv_layers_kernel_sizes)
        if self.predict_potential_function and layer == num_layers - 1:
            assert len(padding_layers) == 0, padding_layers
            if self.config.conv_transpose:
                # Meaning of `padding` in `ConvTranspose` is negated.
                # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                padding -= 1
                conv_kwargs = {**conv_kwargs, 'output_padding': 1}
            else:
                padding += 1
        return padding, padding_mode, padding_layers, conv_kwargs


class _TakeCenterZSlices:
    __slots__ = ('pad',)
    def __init__(self, pad: int):
        self.pad = pad

    def __repr__(self):
        return f'{self.__class__.__name__}(pad={self.pad})'

    def __call__(self, x: TensorCollection):
        def _slice(layer: np.ndarray):
            shape = layer.shape
            assert len(shape) == 5, shape  # (1, 3 or 6, z, y, x)

            center = shape[2] // 2
            begin = center - self.pad
            end = center + self.pad + 1
            return layer[:, :, begin:end, :, :]

        return x.map(_slice)


class TrajectoryRecorder(adaled.TrajectoryRecorder):
    """Record optionally full resolution of the micro solver."""
    def __init__(self,
                 config: CombinedConfigBase,
                 generator: adaled.AdaLEDGenerator,
                 unweighted_loss: ReconstructionLoss,
                 extra_sim_records: Dict[str, adaled.DynamicArray] = {},
                 **kwargs):
        self.generator = generator
        self.micro: CUPSolverBase = generator.micro
        self.record_micro_qoi = adaled.DynamicArray()
        self.record_validation = adaled.DynamicArray()
        extra_sim_records = {
            **extra_sim_records,
            # No need to record x_macro_qoi, it is already recorded with z.
            'x_micro_qoi': self.record_micro_qoi,
            'validation': self.record_validation,
        }
        if config.micro.record_full_resolution:
            extra_sim_records['x_micro_full_resolution'] = \
                    self.record_full_resolution = adaled.DynamicArray()
        else:
            self.record_full_resolution = None

        self.cmp_error = generator.parent.criteria.error_func
        self.unweighted_loss = unweighted_loss

        super().__init__(config=config.led.recorder,
                         extra_sim_records=extra_sim_records, **kwargs)

    def record_step(self, i: int, relative_i: int, step: AdaLEDStep, *args, **kwargs):
        super().record_step(i, relative_i, step, *args, **kwargs)
        config = self.config
        if self.record_full_resolution is not None and relative_i % config.x_every == 0:
            x: Optional[CUPTensorCollection] = step.x
            if x is not None:
                full = x.full_resolution_state
                # This array is huge, better reserve in advance.
                self.record_full_resolution.reserve(
                        config.num_steps // config.x_every,
                        hint_elements=full[np.newaxis])
                self.record_full_resolution.append(full)
            else:
                self.record_full_resolution.append(np.nan)

        if step.x is not None:
            # With for_state=False, possibly more QoIs are included, compared
            # to what is returned by get_state and stored in step.x.
            qoi = self.micro.get_quantities_of_interest(for_state=False)
            if qoi.keys():
                if not isinstance(qoi, TensorCollection):
                    qoi = TensorCollection(qoi)
                self.record_micro_qoi.append(qoi)
        else:
            # At this point, shape must be known.
            # TODO: If QoI is empty, no nans should be added.
            self.record_micro_qoi.append(np.nan)

        if self.generator.parent.config.always_run_micro:
            with self.generator.measure_stats_overhead():
                if step.z is not None:
                    validation = self.compute_validation_stats(step)
                else:
                    validation = np.nan
                self.record_validation.append(validation)

    def compute_validation_stats(self, step: AdaLEDStep) -> TensorCollection:
        """Compute validation errors."""
        transformer = self.generator.parent.transformer
        x_micro = step.x
        with torch.no_grad():
            x_macro = transformer.inverse_transform(step.z)
            cmp_error = self.cmp_error(x_macro, x_micro)
            loss = self.unweighted_loss(x_macro, x_micro)
        return TensorCollection(cmp_error=cmp_error, unweighted_loss=loss)


class TrajectoryRecorder2D(TrajectoryRecorder):
    pass


class TrajectoryRecorder3D(TrajectoryRecorder):
    def __init__(self, config: CombinedConfigBase,
                 generator: adaled.AdaLEDGenerator, **kwargs):
        # Micro needs padding of +-1 in case vorticity is not stored.
        pad_micro = 1 if not config.micro.vorticity_in_state else 0

        # Macro needs +-2 if output is vector potential, +-1 if output is v.
        pad_macro = 2 if config.micro.predict_potential_function else 1

        config.led.recorder.transform_x_micro = _TakeCenterZSlices(pad_micro)
        config.led.recorder.transform_x_macro = _TakeCenterZSlices(pad_macro)
        super().__init__(config, generator, **kwargs)


class SetupBase:
    MOVIE_SCRIPT_PATH: Optional[str] = None
    PLOTTING_SCRIPT_PATH: Optional[str] = None
    POSTPROCESS_SCRIPT_PATH: Optional[str] = None

    def __init__(self, config: CombinedConfigBase):
        self.config = config

    def make_criteria(self, mr: Multiresolution):
        if self.config.led.criteria.use_full_resolution:
            error_func = FullResolutionComparisonError(self.config, mr)
        else:
            error_func = self.make_loss(mr)

        return adaled.SimpleCriteria(
                self.config.led.criteria, error_func=error_func)

    def make_micro(self, F_func, comm) -> CUPSolverBase:
        raise NotImplementedError()

    def make_recorder(self,
                      generator: adaled.AdaLEDGenerator,
                      loss: ReconstructionLoss) -> adaled.TrajectoryRecorder:
        raise NotImplementedError()

    def make_loss(self, mr: Multiresolution, **kwargs) -> ReconstructionLoss:
        return ReconstructionLoss(self.config, mr, **kwargs)

    def make_datasets(self, comm: 'MPI.Intracomm'):
        dataset_config = self.config.dataset
        led_trainer_config = self.config.led.led_trainer
        def wrap(dataset, global_capacity: int, check_batch: bool):
            assert global_capacity % comm.size == 0, (global_capacity, comm.size)
            local_capacity = global_capacity // comm.size
            if check_batch:
                # Once the dataset is full, we want to have an integer number of
                # batches, otherwise the effective batch size drops, from the
                # mathematical, not from the performance point of view.
                assert local_capacity % led_trainer_config.transformer_batch_size == 0, \
                       (local_capacity, led_trainer_config.transformer_batch_size)
                assert local_capacity % led_trainer_config.macro_batch_size == 0, \
                       (local_capacity, led_trainer_config.macro_batch_size)

            datasets = adaled.utils.data.datasets
            policy = adaled.utils.data.replacement.RandomReplacementPolicy()
            dataset = datasets.CappedTrajectoryDataset(dataset, policy, local_capacity)
            dataset = datasets.FixedLengthTrajectoryDataset(
                    dataset, dataset_config.trajectory_length)
            return dataset

        if dataset_config.storage == 'hdf5':
            from adaled.utils.data.hdf5 import HDF5DynamicTrajectoryDataset as HDF5
            train_dataset = HDF5('dataset/train/', 'dataset.h5', comm)
            valid_dataset = HDF5('dataset/valid/', 'dataset.h5', comm)
        else:
            from adaled.utils.data.datasets import UniformInMemoryTrajectoryDataset as DS
            train_dataset = DS()
            valid_dataset = DS()

        import adaled.utils.data.collections as data_collections
        train_dataset = wrap(train_dataset, dataset_config.train_capacity, check_batch=True)
        valid_dataset = wrap(valid_dataset, dataset_config.valid_capacity, check_batch=False)
        total_capacity = dataset_config.train_capacity + dataset_config.valid_capacity
        trajectory_datasets = data_collections.DynamicTrajectoryDatasetCollection(
                train_dataset, valid_dataset,
                train_portion=dataset_config.train_capacity / total_capacity)
        return trajectory_datasets

    def make_transformer_hierarchy(self, mr: Multiresolution) \
            -> 'RecDict[str, Tuple[Transformer, int]]':
        config = self.config
        if config.macro.use_conv_rnn:
            return adaled.IdentityTransformer()
        mr.describe()
        # The number of channels in the input and in the output are different
        # because we do not predict the vorticity, hence the custom
        # inv_x_scaling. Input vorticity is used for the curl(output_v) loss.
        scaling = config.make_scaling_x()
        inv_scaling = config.make_inv_scaling_x()
        print("Transform x-scaling:", scaling)
        print("Transform output scaling:", inv_scaling)
        layer_aes = {}
        for i, (enc, layer) in enumerate(zip(config.autoencoder.encoders, mr.layers)):
            enc.input_shape = (config.get_num_ae_input_channels(),) + layer.downscaled_shape
            encoder = adaled.ConvMLPEncoder(enc)
            decoder = ConvDecoder(config, encoder)
            print(f"Encoder and decoder for layer #{i}:")
            encoder.describe()
            decoder.describe()
            if config.autoencoder.vorticity_in_encoder and \
                    not config.micro.vorticity_in_state:
                if config.micro.predict_pressure:
                    raise NotImplementedError("ConcatVorticity layers don't support pressure")
                h = config.micro.compute_h_for_layer(i)
                if len(config.micro.cells) == 3:
                    vorticity_layer = [ConcatVorticity3DLayer(h)]
                else:
                    vorticity_layer = [ConcatVorticity2DLayer(h)]
            else:
                vorticity_layer = []

            encoder = torch.nn.Sequential(*vorticity_layer, scaling.to_torch(), encoder)
            decoder = torch.nn.Sequential(decoder, inv_scaling.to_torch())
            ae = adaled.AutoencoderTransformer(encoder, decoder)
            layer_aes[f'layer{i}'] = (ae, enc.latent_state_dim)
        hierarchy = {'layers': layer_aes}
        return hierarchy

    def make_transformer(self, mr: Multiresolution):
        hierarchy = self.make_transformer_hierarchy(mr)
        return adaled.CompoundAutoencoder(
                hierarchy, backend=adaled.get_backend(torch.arange(5)))

    def make_macro_solver_and_trainer(self, *, trainer: bool = True):
        micro_config = self.config.micro
        macro_config = self.config.macro
        max_v = self.config.compute_max_v()
        h = micro_config.extent / np.max(micro_config.cells)
        shift_per_timestep = max_v * micro_config.dt_macro
        cells_per_timestep = shift_per_timestep / h
        print(f"Estimated flow speed:\n"
              f"    {max_v}\n"
              f"    {shift_per_timestep} 1/timestep\n"
              f"    {cells_per_timestep} cells/timestep\n"
              f"    {micro_config.cells[0] / cells_per_timestep} timesteps/x-extent")
        if macro_config.use_conv_rnn:
            kernel_range = macro_config.conv_rnn.total_kernel_range()
            print(f"Convolutional RNN kernel range: {kernel_range}", flush=True)
            if kernel_range < cells_per_timestep:
                raise ValueError(
                        f"ConvRNN kernel range of {kernel_range} cells "
                        f"is too narrow for the maximum velocity of "
                        f"({cells_per_timestep} cells/timestep)")
            rnns = [macro_config.conv_rnn.make() for _ in range(macro_config.ensemble_size)]
        else:
            rnns = [adaled.RNN(macro_config.rnn) for _ in range(macro_config.ensemble_size)]

        adaled.print_model_info(rnns[0])

        if not macro_config.use_conv_rnn and macro_config.rnn.has_sigma2:
            ensemble = adaled.ProbabilisticPropagatorEnsemble(rnns)
            kwargs = {
                'loss': adaled.ProbabilisticLosses(macro_config.nll_weight),
            }
            if macro_config.adversarial_eps > 0:
                kwargs['trainer_cls'] = adaled.RNNTrainerWithAdversarialLoss
                kwargs['adversarial_eps'] = macro_config.adversarial_eps
                kwargs['adversarial_loss'] = adaled.ProbabilisticLosses(
                        macro_config.nll_weight, mse_weight=None)
            else:
                kwargs['trainer_cls'] = adaled.RNNTrainer
        else:
            ensemble = adaled.DeterministicPropagatorEnsemble(rnns)
            kwargs = {}

        if trainer:
            trainer = ensemble.make_trainer(macro_config.training, **kwargs)
            return (ensemble, trainer)
        else:
            return ensemble

    def main_server(self, topology: Topology):
        config = self.config
        macro, ensemble_trainer = self.make_macro_solver_and_trainer()
        mr = config.micro.make_multiresolution()
        transformer = self.make_transformer(mr)
        if not config.macro.use_conv_rnn:
            transformer_trainer = adaled.AutoencoderTrainer(
                    transformer.model,
                    **config.autoencoder.training.make(transformer.model.parameters()),
                    loss=self.make_loss(mr))
        else:
            transformer_trainer = None
        datasets = self.make_datasets(topology.server_comm)

        led_trainer = config.led.led_trainer.make(
                macro_trainer=ensemble_trainer,
                transformer=transformer,
                transformer_trainer=transformer_trainer)

        metadata = {
            'rank_topology': topology.specification,
            'num_clients': topology.active_intercomm.remote_size,
            'num_params.transformer': \
                    adaled.estimate_model_num_params(transformer.model),
            'num_params.per_rnn': \
                    adaled.estimate_model_num_params(macro.propagators[0]),
        }
        adaled.save(metadata, 'metadata.json')

        led = adaled.AdaLEDServer(
                config.led.server, macro, datasets, trainer=led_trainer,
                transformer=transformer, topology=topology)
        led.diagnostics.add_dataset_histograms(config.led.dataset_histograms)
        # import pudb; pu.db
        led.run()

    def main_client(self, topology: Topology):
        config = self.config
        # Micro and solvers specification.
        F_func = config.make_F_func()
        micro = self.make_micro(F_func, topology.component_comm)
        if not topology.is_adaled_active:
            micro.run_non_root()
            return

        mr = config.micro.make_multiresolution()
        macro = self.make_macro_solver_and_trainer(trainer=False)
        transformer = self.make_transformer(mr)
        criteria = self.make_criteria(mr)

        # Pass external forcing to AdaLED for postprocessing purposes, the network
        # will ignore it depending on the `append_F` config.
        external_forcing_func = adaled.utils.misc.function_to_generator(
                lambda t: np.asarray(F_func(t)), config.micro.dt_macro)

        led = adaled.AdaLEDClient(
                config.led.client, macro, criteria, transformer=transformer,
                topology=topology)
        generator = led.make_generator(micro, external_forcing_func)
        recorder = self.make_recorder(generator, self.make_loss(mr, unweighted=True))
        led.run(generator, recorder)

    def main(self, topology: Topology):
        print(platform.node())
        config = self.config
        config.fix()
        config.validate()
        if topology.world.rank == 0:
            self.save_config_and_scripts()

        if topology.is_adaled_active:
            adaled.init_torch(server_comm=topology.server_comm,
                              double_precision=self.config.micro.double_precision)

        if topology.is_server:
            self.main_server(topology)
        else:
            self.main_client(topology)

    def main_raw_client(self, world: 'MPI.Intracomm'):
        """Run only Cubism, without the overhead of AdaLED."""
        print(platform.node())
        config = self.config
        config.fix()
        config.validate()
        if world.rank == 0:
            self.save_config_and_scripts()

        F_func = config.make_F_func()
        micro = self.make_micro(F_func, world)
        if world.rank != 0:
            micro.run_non_root()
            return

        N = config.led.client.max_steps
        F = F_func(config.micro.dt_macro * np.arange(N))[:, 0]
        if len(F[0].ravel()) == 1:
            F_header = 'F'
        else:
            F_header = ','.join(f'F{k}' for k in range(len(F[0].ravel())))
        with open('raw_client_benchmark.csv', 'w') as f:
            f.write(f'execution_time,{F_header}\n')
            for i in range(N):
                t0 = time.time()
                micro.advance(no_adaled=True)
                execution_time = time.time() - t0
                F_str = ','.join(str(f) for f in F[i].ravel())
                f.write(f'{execution_time},{F_str}\n')
                if i % 1000 == 0:
                    f.flush()

    def main_postprocess_run(self, world: 'MPI.Intracomm', dir: str = '.'):
        """Run only Cubism to fill the NaN values of a record. Stores the
        result in a separate file. So far only one parallel simulation is
        supported."""
        from .micro import MicroStateType
        config = self.config
        config.fix()
        config.validate()
        print(' '.join(map(shlex.quote, sys.argv)))

        # TODO: F should be read from the record and not even be a part of the
        #       micro solver!
        F_func = config.make_F_func()
        micro = self.make_micro(F_func, world)

        # PostprocessRunner will provide original x_micro state, not the
        # autoencoder output.
        micro.default_update_state_type = MicroStateType.GET_STATE
        if world.rank != 0:
            micro.run_non_root()
            return

        from .postprocess import PostprocessRunnerEx
        import glob
        input_record_files = glob.glob(os.path.join(dir, 'record-000-0*.h5'))
        runner = PostprocessRunnerEx(micro, config.led.recorder, input_record_files)
        runner.run(verbose=True)
        print("Postprocessing run done.")

    def save_config_and_scripts(self):
        print(' '.join(map(shlex.quote, sys.argv)))
        adaled.save(self.config, 'config.json')
        adaled.save(self.config, 'config.pt')
        path = self.MOVIE_SCRIPT_PATH
        if path:
            adaled.save_executable_script(
                    'movie.sh', MOVIE_SCRIPT_TEMPLATE.format(path=path))
        path = self.PLOTTING_SCRIPT_PATH
        if path:
            adaled.save_executable_script(
                    'plot.sh', PLOTTING_SCRIPT_TEMPLATE.format(path=path))
        path = self.POSTPROCESS_SCRIPT_PATH
        if path:
            adaled.save_executable_script(
                    'postprocess.sh', POSTPROCESS_SCRIPT_TEMPLATE.format(path=path))


class SetupBase2D(SetupBase):
    def make_recorder(self, generator: adaled.AdaLEDGenerator, loss: ReconstructionLoss):
        return TrajectoryRecorder2D(self.config, generator, loss)


class SetupBase3D(SetupBase):
    def make_recorder(self, generator: adaled.AdaLEDGenerator, loss: ReconstructionLoss):
        return TrajectoryRecorder3D(self.config, generator, loss)


class ServerClientMainEx(ServerClientMain):
    def add_arguments(self, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        add = parser.add_mutually_exclusive_group().add_argument
        add('--no-adaled', action='store_true', default=False,
            help="disable AdaLED completely, run only the client with no diagnostics")
        # TODO: Remove --postprocess-run, we have now the option
        # led.server.compensate_for_stats_overhead=1.
        add('--postprocess-run', action='store_true', default=None,
            help="Run only the micro solver to fill NaN frames in records. "
                 "Used as a replacement for led.client.always_run_micro=1 "
                 "which would affect the performance of the surrogate. "
                 "Works only when led.recorder.x_every=1.")

    def main(self, argv: Optional[List[str]], config_cls, setup_cls):
        config: CombinedConfigBase = config_cls()
        self.parse_and_process(argv, config)

        setup: SetupBase = setup_cls(config)
        if self.args.no_adaled:
            setup.main_raw_client(self.world)
        elif self.args.postprocess_run:
            setup.main_postprocess_run(self.world)
        else:
            topology = self.init_topology(self.world)
            setup.main(topology)
