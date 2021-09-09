from adaled import AdaLEDStage as Stage, TensorCollection
from adaled.plotting.base import Plotter, Task
from adaled.plotting.plots import Axes, Plot, mpl
from adaled.plotting.utils import divider_colorbar
from adaled.postprocessing.record import LazyRecordLoader

from matplotlib import ticker
import numpy as np

from typing import Any, Dict, Optional, Sequence


class ChannelComparison2DAnimationPlot(Plot):
    def __init__(self,
                 axes: Sequence[Sequence[Axes]],
                 loader: LazyRecordLoader,
                 kwargs: Dict[str, Sequence[Any]] = {},
                 kwargs_matrix: Dict[str, Sequence[Sequence[Any]]] = {},
                 channel_names: Optional[Sequence[str]] = None,
                 pow2_axes: bool = True,
                 *,
                 no_micro_if_macro: bool = False,
                 nan_to_num: Optional[float] = None,
                 **_kwargs):
        """
        Plot channels comparison in four columns:
            (micro propagator, surrogate model, absolute error, normalized absolute error).

        The normalization is performed per channel.

        Arguments:
            kwargs: dictionary of additional kwargs to pass to `imshow`, one
                    for each channel
            kwargs_matrix:
                dictionary of additional kwargs to pass to `imshow`, one for
                each channel AND column (e.g. for 2-channel plots,
                kwargs0123['vmin'] would be a 2x3 matrix)
            no_micro_if_macro:
                Fill micro plot with NaNs during macro-only stage.
                Useful when always_run_micro=1 is used.
            nan_to_num:
                Optionally replace nan values with a given number.
        """
        super().__init__(axes, **_kwargs)

        axes: Sequence[Sequence[Axes]] = self.ax
        num_channels = len(axes)
        assert len(axes[0]) == 3  # (micro, macro, error, normalized error)

        x_data = loader.get_frame_x(0, 0)['x']
        stages = loader.small_fields['metadata', 'stage']

        if pow2_axes:
            shape = x_data['micro'][0].shape
            xlocator = ticker.FixedLocator(np.arange(0, shape[1] + shape[1] // 4, shape[1] // 4))
            ylocator = ticker.FixedLocator(np.arange(0, shape[0] + shape[0] // 4, shape[0] // 4))
        im = [[None, None, None] for _ in range(num_channels)]
        for i, ax in enumerate(axes):
            kw = {key: value[i] for key, value in kwargs.items()}

            def kws(j, **kwargs):
                kwargs.update(kw)
                kwargs.update({k: v[i][j] for k, v in kwargs_matrix.items()})
                kwargs['cmap'] = cmap = mpl.cm.get_cmap(kwargs.get('cmap'))
                cmap.set_bad(color='white')  # nan color
                return kwargs

            x_micro = x_data['micro'][i]
            x_macro = x_data['macro'][i]
            error = np.abs(x_macro - x_micro)
            # yellow in case of transparency (for multiresolution), but nan is
            # still set to white to avoid flashing yellow color.
            ax[0].set_facecolor('yellow')
            ax[1].set_facecolor('yellow')
            ax[2].set_facecolor('yellow')
            im[i][0] = ax[0].imshow(x_micro, **kws(0))
            im[i][1] = ax[1].imshow(x_macro, **kws(1))
            im[i][2] = ax[2].imshow(error, **kws(2, cmap='Reds'))
            divider_colorbar(self.fig, im[i][0], ax=ax[0])
            divider_colorbar(self.fig, im[i][1], ax=ax[1])
            divider_colorbar(self.fig, im[i][2], ax=ax[2])

            if pow2_axes:
                ax[0].xaxis.set_major_locator(xlocator)
                ax[1].xaxis.set_major_locator(xlocator)
                ax[2].xaxis.set_major_locator(xlocator)
                ax[0].yaxis.set_major_locator(ylocator)
                ax[1].yaxis.set_major_locator(ylocator)
                ax[2].yaxis.set_major_locator(ylocator)
                # To ensure the last tick is not removed.
                ax[0].set_xlim(0, shape[1])
                ax[1].set_xlim(0, shape[1])
                ax[2].set_xlim(0, shape[1])
                ax[0].set_ylim(0, shape[0])
                ax[1].set_ylim(0, shape[0])
                ax[2].set_ylim(0, shape[0])
            if i != num_channels - 1:
                ax[0].xaxis.set_ticklabels([])
                ax[1].xaxis.set_ticklabels([])
                ax[2].xaxis.set_ticklabels([])
            if channel_names:
                ax[0].set_ylabel(channel_names[i])
            ax[1].yaxis.set_ticklabels([])
            ax[2].yaxis.set_ticklabels([])

        axes[0, 0].set_title("micro propagator")
        axes[0, 1].set_title("surrogate")
        axes[0, 2].set_title("absolute error")

        def update_func(frame: int):
            x_data = loader.get_frame_x(frame, 0)['x']
            for i in range(num_channels):
                x_micro = x_data['micro'][i]
                x_macro = x_data['macro'][i]
                error = np.abs(x_macro - x_micro)
                if no_micro_if_macro and stages[frame] == Stage.MACRO:
                    x_micro = np.full_like(x_micro, np.nan)
                if nan_to_num is not None:
                    x_micro = np.nan_to_num(x_micro, nan_to_num)
                    x_macro = np.nan_to_num(x_macro, nan_to_num)
                    error = np.nan_to_num(error, nan_to_num)
                im[i][0].set_data(x_micro)
                im[i][1].set_data(x_macro)
                im[i][2].set_data(error)

            return sum(im, [])

        self.update = update_func
