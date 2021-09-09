import adaled
from adaled.transformers.multiresolution import MRLayerConfig, Multiresolution
from adaled.utils.dataclasses_ import DataclassMixin, SPECIFIED_LATER, dataclass, field

import numpy as np

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import copy
import math


@dataclass
class SymmetryBreakingConfig(DataclassMixin):
    """Configuration of artificial oscillations of an obstacle.
    Used to break the flow symmetry.

    The oscillation is active only between time `T0` and `T1`, in which it
    makes `num_periods` sinusodial periods of amplitude `dv`.
    """
    T0: float = 0.0
    T1: float = 1.0
    num_periods: int = 2
    dv: Tuple[float, ...] = (0.0, 0.0, 0.0)

    def modify_velocity(self, v: Tuple[float, ...], t: float) \
            -> Tuple[float, ...]:
        """Apply the oscillation on the given velocity."""
        if self.T0 <= t <= self.T1:
            phase = 2 * math.pi * (self.num_periods * (t - self.T0) / (self.T1 - self.T0))
            v = [v0 + math.sin(phase) * dv for v0, dv in zip(v, self.dv)]
            return v
        else:
            return v


@dataclass
class MicroConfigBase(DataclassMixin):
    """CubismUP2D/3D config base dataclass.

    Arguments:
        profile: (bool) enable profiling of root rank's advance() function,
                 disabled by default
        rampup_steps: (int) number of steps to perform before the first adaled step
        vorticity_in_state: (bool) whether the vorticity is stored directly in
                the state and in the dataset
        enforce_obstacle_velocity: (bool) see compute_obstacle_interior_mask
        enforce_obstacle_velocity_sdf: (float) see compute_obstacle_interior_mask
    """
    cells: Tuple[int, ...] = SPECIFIED_LATER
    min_cells_x: int = 64
    extent: float = 1.0
    nu: float = 1e-4
    cfl: float = 0.4
    dt_macro: float = 0.005
    rampup_steps: int = 0

    extra_cubism_argv: List[str] = field(list)
    mock: bool = False
    profile: bool = False
    output_dir = 'cubismup'
    verbose: bool = False
    double_precision: bool = False

    multiresolution: List[MRLayerConfig] = field(lambda: [MRLayerConfig()])
    pressure_in_state: bool = False
    vorticity_in_state: bool = False

    # Not strictly related to CubismUP, but is related to macro -> micro
    # transition, in order to be able to visualize the potential function
    # before taking derivatives to compute the velocity field.
    predict_potential_function: bool = True

    # Record *full* uniform resolution of the solver as
    # fields/simulations/x_micro_full_resolution at recorder.x_every frequency.
    record_full_resolution: bool = False

    enforce_obstacle_velocity: bool = False
    enforce_obstacle_velocity_sdf: float = 0.0

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'enforce_obstacle_velocity' not in state:
            self.enforce_obstacle_velocity = False
            self.enforce_obstacle_velocity_sdf = 0.0

    @property
    def predict_pressure(self) -> bool:
        # Predicting pressure implies having pressure in state, in order to be
        # able to train the prediction. Having a pressure in state implies the
        # prediction. Otherwise, get_state() right after update_state() would
        # return an outdated pressure. This second requirement could be removed
        # if CUP2D/3D exported a function to compute the pressure.
        return self.pressure_in_state

    def compute_cell_centers(self) -> Tuple[Sequence[float], ...]:
        """Return the x and y (and z in 3D) cell center points."""
        h = self.compute_h()
        return tuple(h * (0.5 + np.arange(cells)) for cells in self.cells)

    def compute_h(self) -> float:
        return self.extent / max(self.cells)

    def compute_h_for_layer(self, layer: int):
        """Compute cell size of a layer. Assumes all cells are cubes."""
        stride = self.multiresolution[layer].stride
        if isinstance(stride, tuple):
            stride = stride[0]
        return self.extent / max(self.cells) * stride

    def compute_hs(self) -> List[float]:
        """Compute h for each layer."""
        return [self.compute_h_for_layer(layer)
                for layer in range(len(self.multiresolution))]

    def compute_obstacle_interior_mask(self) -> np.ndarray:
        """Compute the mask of cells that are inside the obstacles, with
        obstacles expanded or contracted by `enforce_obstacle_velocity_sdf`.

        Cells within the obstacles will be ignored in the autoencoder's
        reconstruction loss and the surrogate's validation error.
        Instead, the velocity will be hard-coded to zero.

        Used only when `enforce_obstacle_velocity` is `True`.
        Note that this feature supports only fixed obstacles.

        Outputs:
            mask: A Boolean np.ndarray of shape `cells[::-1]`.
                  `True` elements denote that this cell should be hardcoded
                  to zero and removed from the autoencoder's loss function.
        """
        raise NotImplementedError(
                "compute_obstacle_interior_mask not implemented for this "
                "case, enforce_obstacle_velocity=1 not supported")

    def compute_nlevels_and_cup_cells(self) -> Tuple[int, Tuple[int, ...]]:
        """Return the nlevels and cells parameters passed to CUP, where cells
        denotes the coarsest-level number of cells, not the finest-level as in
        the Config dataclass.
        """
        cells, min_cells_x = self.cells, self.min_cells_x
        factor = cells[0] // min_cells_x
        nlevels = 1 + int(np.log2(factor))
        assert cells[0] % min_cells_x == 0 and 2 ** (nlevels - 1) == factor, \
               (cells, min_cells_x)
        cup_cells = tuple(c // factor for c in cells)
        return nlevels, cup_cells

    def fix(self, round_sizes_to: List[int]):
        for layer, round_size_to in zip(self.multiresolution, round_sizes_to):
            if layer.round_size_to is None:
                layer.round_size_to = round_size_to
            else:
                assert layer.round_size_to % round_size_to == 0

    def get_dtype_np(self) -> type:
        return np.float64 if self.double_precision else np.float32

    def get_num_export_channels(self) -> int:
        """Return the number of channels returned by micro's get_state()."""
        ndim = len(self.cells)
        out = ndim
        if self.vorticity_in_state:
            out += (3 if ndim == 3 else 1)
        if self.pressure_in_state:
            out += 1
        return out

    def make_multiresolution(self, **kwargs):
        try:
            self.validate()
        except:
            raise Exception("Did you call .fix()?")
        return Multiresolution(self.multiresolution, self.cells, **kwargs)

    def record_qoi_to_state_qoi(
            self, record_qoi: adaled.TensorCollection) -> adaled.TensorCollection:
        """Convert `for_state == False` to `for_state == True` QoI.
        See `Micro.get_quantities_of_interest`."""
        return record_qoi


@dataclass
class Micro2DConfigBase(MicroConfigBase):
    """CubismUP2D config dataclass."""
    # Overriden values.
    cells: Tuple[int, int] = (256, 128)
    extra_cubism_argv: List[str] = \
            field(lambda: '-profilerFreq 1000 -Ctol 1.0 -Rtol 5.0'.split())

    # Extra values.
    mute_all: bool = True


@dataclass
class Micro3DConfigBase(MicroConfigBase):
    """CubismUP3D config dataclass."""
    # Overriden values.
    cells: Tuple[int, int, int] = (256, 128, 128)
    output_dir: str = '.'
    rampup_steps: int = 100
    extra_cubism_argv: List[str] = field(lambda: ["-freqProfiler", "100"])


@dataclass
class LayerLossConfig(DataclassMixin):
    """Loss weights and layer loss definitions.

    Arguments:
        divergence_weight: (float) multiplicative factor for the divergence
                loss, by default 0 because by default we predict the potential
                function, making divergence 0 by definition
    """
    layer_margin_cells: int = 8
    vorticity_weight: float = 0.03
    derivatives_weight: float = 0.0
    # divergence_weight: float = 1e-7
    divergence_weight: float = 0.0
    pressure_weight: float = 1e-1

    # mean(v^2)=0.04 for cyl2d Re=500 layer 0 and ~0.08 for layer 1
    # mean((dv/dx)^2)=160 for cyl2d Re=500 layer 0 and ~1100 for layer 1
    # mean(|w|)=2.90 for cyl2d Re=500 layer 0 and ~9.00 for layer 1
    # mean(p^2)=0.008 for cyl2d Re=500 layer 0 and ~0.025 for layer 1
    relative_loss: bool = True
    relative_derivatives_eps: float = 40.0
    relative_velocity_eps: float = 0.01
    relative_vorticity_eps: float = 0.7
    relative_pressure_eps: float = 0.002

    def unweighted(self) -> 'LayerLossConfig':
        """Return the copy of the config with all weights set to 1.0.

        Note: relative divergence loss is currently unsupported, so
        `divergence_weight` is left as is.
        """
        out = copy.copy(self)
        out.vorticity_weight = 1.0
        out.derivatives_weight = 1.0
        # out.divergence_weight = 1.0
        out.pressure_weight = 1.0
        return out


@dataclass
class DatasetConfig(DataclassMixin):
    """
    Attributes:
        train_capacity: (int) total training capacity on all server ranks combined
        valid_capacity: (int) total validation capacity on all server ranks combined
        trajectory_length: (int) macro steps per trajectory
        storage: (str) hdf5 or memory

    Note: `train_capacity` must be a multiple of `batch size * dataset training
    fraction`, in order to ensure that once the dataset is full that indeed
    full batches are always used.
    """
    train_capacity: int = SPECIFIED_LATER
    valid_capacity: int = SPECIFIED_LATER
    trajectory_length: int = 17  # 1 + 4 + 12
    storage: str = 'hdf5'

    def validate(self, prefix: Optional[str] = None):
        super().validate(prefix)
        assert self.storage in ('hdf5', 'memory'), self.storage


@dataclass
class ScalingConfig(DataclassMixin):
    """
    Attributes:
        v: (optional, float) scaling for velocity, by default equal to
                v_margin * max_v
        vort: (optional, float) scaling for vorticity, by default equal
                to vort_margin * max_v / L
        potential: (optional, float) scaling for potential function, by default equal
                to potential_margin * max_v * L
        pressure: (optional, float) scaling for pressure function, by default
                equal to pressure_margin * max_v^2
        v_margin: (float) (*)
        vort_margin: (float) (*)
        potential_margin: (optional, float) (*)
        pressure_margin: (optional, float) (*)

    (*) margin factor used for the scale of the corresponding field, in case it
        the scale is not specified manually
    """
    v: Optional[float] = None
    vort: Optional[float] = None
    potential: Optional[float] = None
    pressure: Optional[float] = None
    v_margin: float = 2.0
    vort_margin: float = 300.0
    potential_margin: Optional[float] = None
    pressure_margin: Optional[float] = None

    def get_scalings(self, max_v: float, L: float) -> Tuple[float, ...]:
        """Return estimated scale for (v, vort, A, p).
        For unavailable scales, NaNs are returned."""
        def get(scale, margin, factor):
            if scale is None:
                if margin is None:
                    scale = np.nan
                else:
                    scale = margin * factor
            return scale

        v = get(self.v, self.v_margin, max_v)
        vort = get(self.vort, self.vort_margin, max_v / L)
        # TODO: Check if we need A or A^-1 scale in the autoencoder.
        potential = get(self.potential, self.potential_margin, max_v * L)
        pressure = get(self.pressure, self.pressure_margin, max_v ** 2)

        return (v,  vort, potential, pressure)


@dataclass
class MacroConfig(DataclassMixin):
    """Specification of the RNNs and the training of RNNs."""
    rnn: adaled.RNNConfig = field(lambda: adaled.RNNConfig(
            input_size=SPECIFIED_LATER,
            rnn_hidden_size=32, rnn_num_layers=2, activation_output='identity',
            has_sigma2=True, residual=True, append_F=True,
            scaling_F=SPECIFIED_LATER))
    ensemble_size: int = 5

    conv_rnn: adaled.ConvRNNConfig = field(lambda: adaled.ConvRNNConfig(
            layer_channels=[8, 16, 16, 2], ndim=2, append_F=True,
            cell_type='lstm', kernel_sizes=[5, 5, 5, 5], dilations=[1, 2, 2, 1]))
    use_conv_rnn: bool = False

    # Note: `patience` and `milestones` refer to validation steps only!
    training: adaled.TrainingConfig = field(lambda: adaled.TrainingConfig(
            scheduler='none',
            # scheduler='plateau', scheduler_kwargs={'patience': 10},
            # scheduler='multistep',
            scheduler_kwargs={'gamma': 0.8, 'milestones': [2, 3, 4, 5, 6, 7, 8]},
            lr=0.001))
    nll_weight: float = 1e-5
    adversarial_eps: float = 0.00


@dataclass
class AutoencoderConfig(DataclassMixin):
    """
    Arguments:
        vorticity_in_encoder: (bool) whether to pass vorticity to the encoder,
                either directly from the dataset or by manually computing it on
                the fly (must be `True` if `vorticity_in_state` is `True`)
    """
    scaling: ScalingConfig = field(ScalingConfig)
    vorticity_in_encoder: bool = False

    # Encoder/decoder CNNs. If only one encoder is specified, it will be used
    # for all multiprecision layers.
    encoders: List[adaled.ConvMLPEncoderConfig] = field(lambda: [adaled.ConvMLPEncoderConfig(
            latent_state_dim=12,
            # Number of layers should be adjusted to the grid size.
            conv_layers_kernel_sizes=5,
            conv_layers_channels=[14, 16, 18, 20],
            conv_layers_strides=1,
            pool_kernel_sizes=2,
            pool_kernel_strides=2,
            padding_mode='replicate',  # or 'none'?
            activation='celu',
            activation_output='tanh',
            batch_norm=False)])

    loss: LayerLossConfig = field(LayerLossConfig)

    # Note: `patience` and `milestones` refer to validation steps only!
    training: adaled.TrainingConfig = field(lambda: adaled.TrainingConfig(
            scheduler='none',
            # scheduler='plateau', scheduler_kwargs={'patience': 10},
            # scheduler='multistep',
            scheduler_kwargs={'gamma': 0.8, 'milestones': [2, 3, 4, 5, 6, 7, 8]},
            lr=0.001))


@dataclass
class CUPCriteriaConfig(adaled.SimpleCriteriaConfig):
    use_full_resolution: bool = True


@dataclass
class CUPAdaLEDConfig(DataclassMixin):
    dataset_histograms: Dict[str, adaled.HistogramStatsConfig] = field(lambda: {
        # 'mse' part removed in ConfigMixin when has_sigma2 == False.
        'latest_loss_macro_mse': adaled.HistogramStatsConfig(
                25, (1e-6, 1e2), ('metadata', 'latest_loss', 'macro', 'mse'), log=True),
        # Hierarchy added later.
        'latest_loss_transformer': adaled.HistogramStatsConfig(
                25, (1e-6, 1e2), ('metadata', 'latest_loss', 'transformer'), log=True)
    })

    led_trainer: adaled.SequentialLEDTrainerConfig = field(lambda: adaled.SequentialLEDTrainerConfig(
            # Batch size per rank.
            macro_batch_size=8,
            transformer_batch_size=2,
            states_count_policy=adaled.SampleCountPolicyConfig(fraction=0.125),
            trajectories_count_policy=adaled.SampleCountPolicyConfig(fraction=0.125)))

    # AdaLED.
    criteria: CUPCriteriaConfig = field(lambda: CUPCriteriaConfig(
            k_warmup=4,
            k_cmp=12,
            max_cmp_error=0.001,   # Relative mean square error in real space!
            max_uncertainty=0.01,  # Mean variance in latent space!
            max_macro_steps=(400, 500),
            max_micro_steps=(9, 14),  # Range chosen such that always 2 short
                                      # trajectories are added to the dataset.
            num_relaxation_steps=0))
    server: adaled.AdaLEDServerConfig = field(lambda: adaled.AdaLEDServerConfig(
            validation_every=20,
            # `every` is in global cycles. Adjust if working with multiple ranks!
            dump_dataset=adaled.DumpConfig(every=10, hardlink=True),
            dump_macro=adaled.DumpConfig(every=10),
            dump_transformer=adaled.DumpConfig(every=10),
            dump_trainers=adaled.DumpConfig(every=10),
            init_output_folder=False,
            max_trains_without_new_data=10))
    client: adaled.AdaLEDClientConfig = field(lambda: adaled.AdaLEDClientConfig(
            # Never log steps, it requires downloading the x state to the CPU.
            log_every=0,
            max_steps=50052))
    recorder: adaled.RecorderConfig = field(lambda: adaled.RecorderConfig(
            start_every=1000, num_steps=1000, every=1, x_every=100, z_every=1,
            posttransform='float32',
            path_fmt='record-{sim_id:03d}-{start_timestep:07d}.h5'))


@dataclass
class CombinedConfigBase(DataclassMixin):
    micro: MicroConfigBase = SPECIFIED_LATER
    dataset: DatasetConfig = field(DatasetConfig)
    led: CUPAdaLEDConfig = field(CUPAdaLEDConfig)
    autoencoder: AutoencoderConfig = field(AutoencoderConfig)
    macro: MacroConfig = field(MacroConfig)

    def fix(self):
        """Fix arguments depending on other arguments."""
        num_layers = len(self.micro.multiresolution)

        # Fix histogram setup.
        dh = self.led.dataset_histograms
        if not self.macro.rnn.has_sigma2:
            # Remove 'mse' part if RNN is not probabilistic.
            if 'latest_loss_macro_mse' in dh:
                dh['latest_loss_macro_mse'].data = ('metadata', 'latest_loss', 'macro')
        if 'latest_loss_transformer' in dh:
            per_layer = {'v': None, 'vorticity': None, 'divergence': None}
            dh['latest_loss_transformer'].hierarchy = \
                    {'layers': {f'layer{i}': per_layer for i in range(num_layers)}}

        # Ensure there is the same number of encoders as multiresolution layers.
        ae = self.autoencoder
        for enc in ae.encoders:
            enc.fix()
        if len(ae.encoders) != num_layers:
            if len(ae.encoders) == 1:
                ae.encoders = ae.encoders * num_layers
            else:
                raise TypeError(
                        f"Specified {len(ae.encoders)} encoders, while "
                        f"there are {num_layers} multiprecision layers. Note "
                        f"that if only one encoder is specified, that the "
                        f"same encoder setup will be used for all layers.")
        round_sizes_to = [enc.total_scale_factor() for enc in ae.encoders]
        self.micro.fix(round_sizes_to)

        # Set up autoencoders.
        mr = self.micro.make_multiresolution()
        for i, (enc, layer) in enumerate(zip(self.autoencoder.encoders, mr.layers)):
            enc.input_shape = (self.get_num_ae_input_channels(),) + layer.downscaled_shape

        # ConvRNN set up (NOT UP TO DATE).
        self.macro.conv_rnn.input_channels = len(self.micro.cells)

        # Fill missing setup for RNNs.
        total_latent_dim = sum(enc.latent_state_dim for enc in ae.encoders) \
                         + self.extra_latent_size()
        rnn = self.macro.rnn
        rnn.input_size = total_latent_dim \
                + int(bool(rnn.append_F)) * self.get_F_dimensionality()
        rnn.output_size = total_latent_dim
        rnn.scaling_F = self.make_scaling_F()

    def validate(self, *args, **kwargs):
        super().validate(*args, **kwargs)
        if self.micro.vorticity_in_state and not self.autoencoder.vorticity_in_encoder:
            raise ValueError("vorticity_in_encoder must be 1 if vorticity_in_state is 1")
        if self.micro.predict_pressure and not self.micro.pressure_in_state:
            raise ValueError("pressure_in_state must be 1 if predict_pressure is 1")
        if self.micro.predict_potential_function \
                and self.autoencoder.scaling.potential is None \
                and self.autoencoder.scaling.potential_margin is None:
            raise ValueError("potential scaling not specified")
        if self.micro.predict_pressure \
                and self.autoencoder.scaling.pressure is None \
                and self.autoencoder.scaling.pressure_margin is None:
            raise ValueError("pressure scaling not specified")

    def get_num_ae_input_channels(self) -> int:
        """Number of autoencoder input channels, not to be confused with the
        number of exported micro state channels."""
        ndim = len(self.micro.cells)
        assert 2 <= ndim <= 3
        out = ndim
        if self.autoencoder.vorticity_in_encoder:
            out += (3 if ndim == 3 else 1)
        if self.micro.pressure_in_state:
            out += 1
        return out

    def get_num_ae_output_channels(self) -> int:
        """Return the number of autoencoder output channels."""
        ndim = len(self.micro.cells)
        assert 2 <= ndim <= 3
        if self.micro.predict_potential_function:
            out = (3 if ndim == 3 else 1)
        else:
            out = ndim
        if self.micro.predict_pressure:
            out += 1
        return out

    def make_scaling_x(self) -> adaled.Scaling:
        v, vort, A, p = self.autoencoder.scaling.get_scalings(
                self.compute_max_v(), self.micro.extent)
        ndim = len(self.micro.cells)
        assert 2 <= ndim <= 3, ndim

        scale = [v] * ndim
        if self.micro.vorticity_in_state:
            scale.extend([vort] * (3 if ndim == 3 else 1))
        if self.micro.pressure_in_state:
            scale.append(p)

        scale = np.array(scale).reshape((-1,) + (1,) * ndim)
        return adaled.Scaling(min=-scale, max=+scale)

    def make_inv_scaling_x(self):
        """Compute scaling of the output layer of the autoencoder."""
        v, vort, A, p = self.autoencoder.scaling.get_scalings(
                self.compute_max_v(), self.micro.extent)
        ndim = len(self.micro.cells)
        assert 2 <= ndim <= 3, ndim

        scale = []
        if self.micro.predict_potential_function:
            scale.extend([A] * (3 if ndim == 3 else 1))
        else:
            scale.extend([1 / v] * ndim)
        if self.micro.predict_pressure:
            scale.append(1 / p)

        scale = np.array(scale).reshape((-1,) + (1,) * ndim)
        return adaled.Scaling(min=-scale, max=+scale)

    def compute_max_v(self) -> float:
        """Estimate of the maximum velocity in the flow, e.g. maximum obstacle
        velocity."""
        raise NotImplementedError()

    # virtual
    def compute_multiresolution_submask_weights(
            self, mr: Multiresolution) -> List[Optional[np.ndarray]]:
        """Compute loss weight matrices, one for each layer.
        A `None` value stands for uniform weight.

        The function may be invoked with a submasked multiresolution object.
        """
        margin = self.autoencoder.loss.layer_margin_cells
        layers = mr.layers
        if len(layers) == 1:
            return [None]  # No weights.
        elif len(layers) == 2:
            alpha1_for0 = layers[1].compute_alpha(
                    layers[0].box, margin=layers[1].alpha_margin_cells + margin)
            alpha1_for0 = layers[0].slice_and_downscale(alpha1_for0)
            alpha1_for1 = layers[1].compute_alpha(
                    layers[1].box, margin=layers[1].alpha_margin_cells - margin)
            weights = [
                1 - alpha1_for0,
                alpha1_for1,
            ]
            return weights
        else:
            raise NotImplementedError("merging logic for more than 2 layers not implemented")

    def compute_multiresolution_weights(
            self, mr: Multiresolution) -> List[Optional[np.ndarray]]:
        """Combines `compute_multiresolution_submask_weights` with the obstacle
        mask if `micro.enforce_obstacle_velocity` is set to `True`.

        Must be invoked with the full multiresolution object."""
        assert mr.ndim == len(self.micro.cells), \
               "if you want submasked multiresolution info, " \
               "check out compute_multiresolution_submask_weights"
        weights = self.compute_multiresolution_submask_weights(mr)
        if self.micro.enforce_obstacle_velocity:
            mask = (~self.micro.compute_obstacle_interior_mask()).astype(np.float32)
            weights = [weight * layer.slice_and_downscale(mask)
                       for weight, layer in zip(weights, mr.layers)]
        return weights

    # virtual
    def extra_latent_size(self) -> int:
        return 0

    # virtual
    def make_F_func(self) -> Callable[[float], Union[np.ndarray, 'torch.Tensor']]:
        raise NotImplementedError()

    # virtual
    def make_scaling_F(self) -> adaled.Scaling:
        raise NotImplementedError()

    def get_F_dimensionality(self) -> int:
        F = self.make_F_func()(0.0)
        F = np.asarray(F)
        return len(F.ravel())
