from adaled.plotting.base import Plotter, Task
from adaled.plotting.plots import Axes, Plot
from adaled.postprocessing.statistics import \
        Channels1DStats, PostprocessedData1D
import adaled.plotting.utils as utils
import adaled.utils.io_ as io_

import numpy as np

from typing import Sequence

class Channel1DStatsComparisonPlot(Plot):
    def __init__(self,
                 path: str,
                 title: str,
                 datasets: Sequence[Channels1DStats],
                 channel: int,
                 xlabel: str,
                 dataset_names: Sequence[str],
                 *,
                 periodic: bool = True,
                 error_wrt: int = 0,  # -1 to disable
                 **kwargs):
        num_datasets = len(datasets)
        super().__init__(path, nrows=num_datasets + 1, ncols=2, suptitle=title,
                         gridspec_kw={'width_ratios': [3, 1]}, **kwargs)
        axs = self.ax

        distributions = [d.histograms[channel].sum(axis=0) for d in datasets]
        distributions = [d / d.sum() for d in distributions]

        means = [d.means[channel] for d in datasets]

        if error_wrt >= 0:
            assert error_wrt < num_datasets
            if num_datasets > 1:
                axl_error = axs[-1, 0].twinx()
                axr_error = axs[-1, 1].twiny()


        colors = utils.get_default_rgba(num_datasets)
        for i, dataset in enumerate(datasets):
            dataset: Channels1DStats

            state_size = dataset.means.shape[1]
            x = np.arange(state_size)
            h = dataset.histograms[channel]  # (state_size, num_bins)
            mean = means[i]
            bin_centers = dataset.histogram_bins[channel]
            bin_centers = 0.5 * (bin_centers[:-1] + bin_centers[1:])

            # Plot histograms on the left.
            r = dataset.histogram_ranges[channel]
            axs[i, 0].imshow(h.T, aspect='auto',
                             extent=[-0.5, state_size - 0.5, *r])
            axs[i, 0].set_ylabel(dataset_names[i])

            # Plot distribution on the right as x=frequency, y=value (bin).
            axs[i, 1].plot(distributions[i], bin_centers, color=colors[i])
            axs[i, 1].set_xlim(0.0, None)
            axs[i, 1].yaxis.tick_right()

            # Plot all distributions on bottom right together.
            axs[-1, 1].plot(distributions[i], bin_centers, color=colors[i])
            if error_wrt >= 0 and i != error_wrt:
                error = distributions[i] - distributions[error_wrt]
                axr_error.plot(error, bin_centers, color=colors[i],
                               linestyle=':', linewidth=1.0)

            # Plot mean on bottom left.
            utils.periodic_line_x(
                    axs[-1, 0], x, mean, label=dataset_names[i],
                    color=colors[i], periodic_linestyle='--',
                    periodic=periodic)
            if error_wrt >= 0 and i != error_wrt:
                error = means[i] - means[error_wrt]
                utils.periodic_line_x(
                        axl_error, x, error, label=dataset_names[i],
                        color=colors[i], linestyle=':', periodic_linestyle=':',
                        periodic=periodic, linewidth=1.0)

        axs[0, 0].set_title("distribution along the axis")
        axs[0, 1].set_title("value distribution")
        axs[-1, 0].legend()
        axs[-1, 0].set_xlabel(xlabel)
        axs[-1, 0].set_ylabel("average")
        if error_wrt >= 0 and num_datasets > 1:
            axl_error.set_ylabel("error")
        if periodic:
            x_extent = (x[-1] - x[0]) * len(x) / (len(x) - 1)
            axs[-1, 0].set_xlim(x[0] - 0.05 * x_extent, x[-1] + 0.05 * x_extent)
        else:
            axs[-1, 0].set_xlim(x[0], x[-1])
        axs[-1, 1].set_xlim(0.0, None)
        axs[-1, 1].set_xlabel("frequency")
        axs[-1, 1].set_ylabel("value")
        if error_wrt >= 0 and num_datasets > 1:
            # axr_error.set_xlabel("error")
            pass

        self.fig.subplots_adjust(wspace=0.3, hspace=0.3)


class PlotPostprocessed1D(Plotter):
    def set_up(self):
        if hasattr(self, 'data'):
            return
        self.data: PostprocessedData1D = io_.load('postprocessed_data_1d.pt')

    def make_path(self, path):
        return path

    def tasks_postprocessed_1d(self):
        data = self.data
        num_channels = len(data.original.means)
        for i in range(num_channels):
            yield Task(Channel1DStatsComparisonPlot,
                       self.make_path(f'channel-stats-{i}.png'),
                       f"Channel #{i} distributions: dataset vs reconstructed\n"
                       f"(averaged over {data.num_samples} states from "
                       f"{data.num_trajectories} trajectories)",
                       [data.original, data.reconstructed], i, xlabel="x",
                       dataset_names=["original", "reconstructed"])


if __name__ == '__main__':
    PlotPostprocessed.main()
