from .loss import LayerLoss
from .micro import AutoencoderReconstructionHelper, MicroStateHelper
from .setup import CombinedConfigBase
from .utils_2d import compute_vorticity_2d, stream_function_to_velocity
from .utils_3d import curl

from adaled import TensorCollection
from adaled.plotting.base import Task, Plotter
from adaled.plotting.plot_diagnostics import DiagnosticsAnimationPlot
from adaled.plotting.plots import Axes, Figure, Plot, mpl, plt
from adaled.plotting.plots_2d import ChannelComparison2DAnimationPlot
from adaled.postprocessing.record import LazyRecordLoader
import adaled

import numpy as np

from typing import Optional, Sequence, Tuple
import argparse
import glob

class LazyRecordLoaderExBase(LazyRecordLoader):
    """Load frames and compute velocity field from stream functions and
    vorticity from velocity on the fly."""
    def __init__(
            self,
            *args,
            config: CombinedConfigBase,
            layer: Optional[int],
            only_vorticity: bool = False,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config  # For derived classes.
        self.mr = config.micro.make_multiresolution()
        self.hs = [layer.stride[0] * config.micro.compute_h() for layer in self.mr.layers]
        self.layer = layer
        self.micro_helper = MicroStateHelper(config.micro)
        self.macro_helper = AutoencoderReconstructionHelper(config.micro)
        self.only_vorticity = only_vorticity

    def get_frame_x(self, *args, **kwargs):
        sim = super().get_frame_x(*args, **kwargs)
        x = sim['x']
        x['micro'] = self.process_layers(x, 'micro')
        x['macro'] = self.process_layers(x, 'macro')
        if self.only_vorticity:
            x = x.map(lambda array: array[2:3, :, :])
        return sim

    def process_layers(self, x: TensorCollection, key: str):
        assert key in ('micro', 'macro'), key
        x = x[key, 'layers']

        if self.layer is not None:
            return self.compute_v_vort_p_2d(self.layer, x[f'layer{self.layer}'], key)
        layers = [self.compute_v_vort_p_2d(i, x[key_], key)
                  for i, key_ in enumerate(sorted(x.keys()))]
        return self.mr.rebuild_unsafe(layers)

    def compute_v_vort_p_2d(self, layer: int, v: np.ndarray, key: str) -> np.ndarray:
        """Compute velocity and vorticity channels, given the micro or macro
        state with optional pressure stripped away."""
        assert key in ('micro', 'macro'), key
        raise NotImplementedError(self)


class LazyRecordLoaderEx2D(LazyRecordLoaderExBase):
    def compute_v_vort_p_2d(self, layer: int, channels: np.ndarray, key: str) -> np.ndarray:
        if key == 'micro':
            return self.micro_helper.layer_to_v_vort_p(layer, channels[np.newaxis])[0]
        elif key == 'macro':
            return self.macro_helper.layer_to_v_vort_p(layer, channels[np.newaxis])[0]
        else:
            raise NotImplementedError(key)


class LazyRecordLoaderEx3D(LazyRecordLoaderExBase):
    def compute_v_vort_p_2d(self, layer: int, channels: np.ndarray, key: str) -> np.ndarray:
        """Compute velocity, vorticity and optionally pressure fields for the
        center z-slice."""
        assert channels.ndim == 4, channels.shape
        num_channels, z, y, x = channels.shape

        # Take only center 5 slices.
        # Slice such that it works for already 3D arrays already sliced to z ==
        # 3 or z == 5. Note: micro needs only +-1 slice, while macro needs + 2.
        # In principle, the data is already sliced across the z-axis anyway.
        if key == 'micro':
            channels = channels[:, z // 2 - 1 : z // 2 + 2, :, :]
            out3d = self.micro_helper.layer_to_v_vort_p(layer, channels[np.newaxis])[0]
        elif key == 'macro':
            channels = channels[:, max(z // 2 - 2, 0) : z // 2 + 3, :, :]
            out3d = self.macro_helper.layer_to_v_vort_p(layer, channels[np.newaxis])[0]

        assert out3d.ndim == 4, out3d.shape          # (channels, z = 3 or 5, y, x)
        out3d = out3d[:, out3d.shape[1] // 2, :, :]  # Center slice.
        if out3d.shape[0] == 3 + 3 + 1:              # v, vort, p
            return out3d[[0, 1, 5, 6], :, :]         # (vx, vy, vort_z, p)
        elif out3d.shape[0] == 3 + 3:                # v, vort
            return out3d[[0, 1, 5], :, :]            # (vx, vy, vort_z)
        else:
            raise NotImplementedError("unreachable", out3d.shape)


class DiagnosticsAnimationPlotEx(DiagnosticsAnimationPlot):
    TRANSFORMER_COLORS = ['black', 'black', 'black', 'gray', 'gray', 'gray']
    TRANSFORMER_LINE_STYLES = ['--', '-.', ':']


def plot_movie_2d(path: str, config: CombinedConfigBase, rank: int, report: bool = False, **kwargs):
    paths = sorted(glob.glob(f'record-{rank:03d}-0*.h5'))
    if not paths:
        print("No records found, skipping the movie (or movie frame) plot.")
        return

    if not report:
        try:
            diagnostics = adaled.load(f'diagnostics-{rank:03d}.pt')
        except FileNotFoundError as e:
            print("WARNING:", str(e))
            diagnostics = None
    else:
        diagnostics = None

    _plot_movie_2d(path, config, paths, diagnostics, report=report, **kwargs)


def _plot_movie_2d(
        path: str,
        config: CombinedConfigBase,
        input_paths: Sequence[str],
        diagnostics: Optional[TensorCollection],
        layer: Optional[int] = None,
        num_frames: int = -1,
        frame_begin: int = 0,
        frame_end: Optional[int] = None,
        frame_skip: Optional[int] = None,
        frames: Optional[Sequence[int]] = None,
        fps: int = 30,
        workers: Optional[int] = None,
        *,
        bitrate: int = -1,
        clean: bool = False,
        dpi: int = 100,
        no_micro_if_macro: bool = False,
        report: bool = False,
        only_vorticity: bool = False,
        loader_cls: type = LazyRecordLoaderExBase):
    loader = loader_cls(input_paths, num_frames, config=config, layer=layer,
                        only_vorticity=only_vorticity)
    fields = loader.small_fields
    metadata = fields['metadata']

    if report:
        fig: Figure = plt.figure(constrained_layout=True, figsize=(12.0, 5.5), dpi=dpi)
        gs = fig.add_gridspec(nrows=1, ncols=1, height_ratios=[4])
    elif not only_vorticity:
        fig: Figure = plt.figure(constrained_layout=True, figsize=(10.5, 8.0), dpi=dpi)
        gs = fig.add_gridspec(nrows=(1 if report else 2), ncols=1, height_ratios=[4, 2])
    else:
        fig: Figure = plt.figure(constrained_layout=True, figsize=(12, 5.0), dpi=dpi)
        gs = fig.add_gridspec(nrows=(1 if report else 2), ncols=1, height_ratios=[2, 2])
    num_channels = (1 if only_vorticity else 3)
    gs_channels = mpl.gridspec.GridSpecFromSubplotSpec(num_channels, 3, subplot_spec=gs[0])
    ax_channels = [[fig.add_subplot(gs_channels[i, j]) for j in range(3)]
                   for i in range(num_channels)]
    # axh: Axes = fig.add_subplot(gs[0, 1])

    # # Top right: dataset histogram.
    # # https://matplotlib.org/stable/gallery/animation/animated_histogram.html
    # F_histogram = diagnostics['dataset_F_histogram']
    # bar_container = axh.bar(
    #         np.linspace(*config.F_histogram_range, config.F_histogram_nbins),
    #         F_histogram[-1])
    # axh.yaxis.set_major_formatter(mpl.ticker.EngFormatter(sep=""))
    # axh.set_xlabel("$\mu$")
    # axh.set_ylim(0, F_histogram.max())
    # axh.set_ylabel("number of states in the training dataset")

    max_v = config.compute_max_v()
    # v_scale = 0.7 * max_v
    # v_scale = 1.2 * max_v
    v_scale = 0.6 * max_v
    # vort_scale = 40.0 * max_v / config.micro.extent
    # vort_scale = 120.0 * max_v / config.micro.extent
    vort_scale = 60.0 * max_v / config.micro.extent
    norm1 = mpl.colors.Normalize(vmin=-v_scale, vmax=+v_scale)
    norm2 = mpl.colors.Normalize(vmin=0.0, vmax=0.8 * v_scale)
    norm3 = mpl.colors.Normalize(vmin=-vort_scale, vmax=+vort_scale)
    norm4 = mpl.colors.Normalize(vmin=0.0, vmax=0.8 * vort_scale)
    norm_abs = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    kwargs_matrix = {
        # [micro, macro, error, abs error]
        'norm': [
            [norm1, norm1, norm2, norm_abs],
            [norm1, norm1, norm2, norm_abs],
            [norm3, norm3, norm4, norm_abs],
        ],
        'cmap': [['RdBu', 'RdBu', 'Reds', 'Reds']] * 3,
    }
    if layer is not None:
        loss = LayerLoss(config, loader.mr)
        weight = loss.layer_loss_weights[layer]
        if weight is not None:
            assert loader.mr.layers[layer].downscaled_shape == weight.shape, \
                   (loader.mr.layers[layer].downscaled_shape, weight.shape)
            if weight.ndim == 3:
                weight = weight[weight.shape[0] // 2, :, :]  # The center z-slice.
            alpha = np.clip(weight, 0, 1)
        else:
            alpha = 1.0
        kwargs_matrix['alpha'] = [[None, alpha, alpha, alpha]] * 3

    channel_names = ["$u_x$", "$u_y$", "vorticity"]
    if only_vorticity:
        kwargs_matrix = {key: value[2:3] for key, value in kwargs_matrix.items()}
        channel_names = channel_names[2:3]

    state_plot = ChannelComparison2DAnimationPlot(
            ax_channels, loader, kwargs_matrix=kwargs_matrix,
            channel_names=channel_names,
            no_micro_if_macro=no_micro_if_macro, nan_to_num=0.0)
    if diagnostics:
        ax1: Axes = fig.add_subplot(gs[1])
        _last_cycle = np.searchsorted(diagnostics['start_timestep'],
                                      metadata['timestep'][-1])
        diagnostics = diagnostics[:_last_cycle + 1]
        if clean:
            del diagnostics['losses']['macro_train']
            del diagnostics['losses']['transformer_train']
            del fields['simulations']['uncertainty']
        diagnostics_plot = DiagnosticsAnimationPlotEx(
                ax1, fields, diagnostics, error_label="MSE", F_label="Re",
                macro_roll_window=500, criteria=config.led.criteria,
                legend=(not clean))
        diagnostics_plot.ax.set_ylim(1e-7 if loader.mr.ndim == 3 else 1e-6, 1e0)
        diagnostics_plot.ax_macro.set_ylim(0.0, 1.0)
        cycle_t = diagnostics['start_timestep']

    def update(frame):
        updated = []

        timestep = metadata['timestep'][frame]

        if diagnostics:
            last_cycle = getattr(update, 'last_cycle', -1)
            update.last_cycle = cycle = np.searchsorted(cycle_t[:-1], timestep)
            updated += diagnostics_plot.update(frame, cycle, last_cycle)
        else:
            cycle = "?"

        updated += state_plot.update(frame)

        # if cycle != last_cycle:
        #     for count, rect in zip(F_histogram[cycle], bar_container.patches):
        #         rect.set_height(count)
        #     updated.extend(bar_container.patches)

        if not report:
            fig.suptitle(f"AdaLED cycle #{cycle}, timestep #{timestep}")

        return updated

    from adaled.plotting.animation import parallelized_animation
    if frame_skip is None:
        frame_skip = config.led.recorder.x_every
        if frame_skip > 1:
            print(f"Plotting only frames for which the real state is "
                  f"available (x_every={frame_skip}).")
    if frames is None:
        frames = range(len(metadata))[frame_begin:frame_end:frame_skip]
    parallelized_animation(path, fig, update, frames,
                           interval=1000 // fps, blit=True, workers=workers,
                           bitrate=bitrate)
    plt.close(fig)
    print(f"Saved {path}")


def plot_movie_2d_micro_vorticity(
        path: str,
        config: CombinedConfigBase,
        rank: int,
        layer: Optional[int] = None,
        num_frames: int = -1,
        frame_begin: int = 0,
        frame_end: Optional[int] = None,
        frame_skip: Optional[int] = None,
        workers: Optional[int] = None,
        *,
        fps: int = 30,
        loader_cls: type = LazyRecordLoaderExBase,
        **kwargs):
    """Make a 2D movie of only vorticity and nothing else."""
    input_paths = sorted(glob.glob(f'record-{rank:03d}-0*.h5'))
    if not input_paths:
        print("No records found, skipping the movie (or movie frame) plot.")
        return

    loader = loader_cls(input_paths, num_frames, config=config, layer=layer)
    x_data = loader.get_frame_x(0, 0)['x']
    vort_micro = x_data['micro'][2]  # Channel #2 should be vorticity.

    # fig, ax = plt.subplots(1, 1, figsize=(12.0, 6.0), frameon=False, tight_layout=True)
    dpi = 128
    height, width = vort_micro.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    max_v = config.compute_max_v()
    scale = 120.0 * max_v / config.micro.extent
    norm = mpl.colors.Normalize(vmin=-scale, vmax=+scale)
    im = ax.imshow(vort_micro, cmap='RdBu', norm=norm)

    def update(frame):
        x_data = loader.get_frame_x(frame, 0)['x']
        vort_micro = x_data['micro'][2]  # Channel #2 should be vorticity.
        im.set_data(vort_micro)
        return [im]

    if num_frames == -1:
        num_frames = len(loader.small_fields['metadata', 'stage'])
    frames = range(num_frames)[frame_begin:frame_end:frame_skip]
    from adaled.plotting.animation import parallelized_animation
    parallelized_animation(path, fig, update, frames,
                           interval=1000 // fps, blit=True, workers=workers)
    plt.close(fig)
    print(f"Saved {path}")


def tasks_all_rank_movies_2d(path_fmt: str, config: CombinedConfigBase,
                             movie_func=plot_movie_2d, **kwargs):
    """
    Arguments:
        path_fmt: (str) path format with a 'rank' slot
    """
    metadata = adaled.load('metadata.json')
    for rank in range(metadata['num_clients']):
        yield Task(movie_func, path_fmt.format(rank=rank), config,
                   rank, **kwargs)


class MoviePlotterBase(Plotter):
    def movie_func(self, *args, **kwargs):
        # FIXME: MoviePlotter sometimes used without properly initializing it
        # (without calling add_arguments).
        only_vorticity = getattr(self.args, 'only_vorticity', 'no')
        if only_vorticity == 'standalone':
            return plot_movie_2d_micro_vorticity(*args, **kwargs)
        else:
            only_vorticity = only_vorticity == 'yes'
            return plot_movie_2d(*args, only_vorticity=only_vorticity, **kwargs)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument
        add('--num-frames', type=int, default=-1)
        add('--slice', type=int, nargs=2, default=(0, None))
        add('--skip', type=int)
        add('--only-vorticity', type=str, choices=('no', 'yes', 'standalone'), default='no')
        add('--fps', type=int, default=30, help="frame rate")
        add('--dpi', type=int, default=100, help="DPI")
        add('--no-micro-if-macro', default=False, action='store_true',
            help="disable micro view on macro-only steps, "
                 "useful for validation runs with always_run_micro=1")
        add('--clean', default=False, action='store_true',
            help="disable some less relevant stats")
        add('--output-prefix', type=str, default='')
        add('--bitrate', type=int, default=-1,
            help="bitrate in kb/s, default -1 (automatic)")

    def tasks_movie(self, layer: Optional[int] = None):
        args = self.args
        config = self.context.load_config()
        suffix = '' if layer is None else f'-layer{layer}'
        yield from tasks_all_rank_movies_2d(
                f'movie{args.output_prefix}-{{rank:03d}}{suffix}.mp4',
                config, movie_func=self.movie_func,
                num_frames=args.num_frames,
                frame_begin=args.slice[0], frame_end=args.slice[1],
                frame_skip=args.skip, workers=args.jobs,
                _task_no_parallel=True, layer=layer, fps=args.fps,
                no_micro_if_macro=args.no_micro_if_macro,
                clean=args.clean, bitrate=args.bitrate, dpi=args.dpi)


class Movie2DPlotter(MoviePlotterBase):
    def movie_func(self, *args, **kwargs):
        return super().movie_func(*args, loader_cls=LazyRecordLoaderEx2D, **kwargs)


class Movie3DPlotter(MoviePlotterBase):
    def movie_func(self, *args, **kwargs):
        return super().movie_func(*args, loader_cls=LazyRecordLoaderEx3D, **kwargs)

    def tasks_movie(self):
        config: CombinedConfigBase = self.context.load_config()
        for layer in range(len(config.micro.multiresolution)):
            yield from super().tasks_movie(layer=layer)
