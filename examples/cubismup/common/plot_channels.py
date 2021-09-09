"""
Plot channels as standalone images.

Usage example:
    python3 -m examples.cubismup.common.plot_channels --frames 100 200 300 --group=fields/simulations/x/micro --path=record-000-0000000.h5 --output-prefix=mr-micro
"""
# Import immediately because cubismup2d is compiled with a newer version of
# OpenMP than the one used by torch.
import cubismup2d as cup2d

from .config import CombinedConfigBase
from .loss import LayerLoss
import adaled

import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import numpy as np

import argparse

def downscale(array: np.ndarray, scale: int):
    shape = array.shape
    assert shape[1] % scale == 0 and shape[2] % scale == 0, (shape, scale)
    array = array.reshape(shape[0], shape[1] // scale, scale, shape[2] // scale, scale)
    array = array.mean(axis=(-1, -3))
    return array


def main():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--frames', type=int, nargs='+', required=True, help="frames to plot")
    add('--group', type=str, required=True)
    add('--path', type=str, required=True, help="path of the HDF5 file")
    add('--output-prefix', type=str, required=True, help="output path prefix")
    add('--downscale', type=int, default=1, help="downscale image")

    config: CombinedConfigBase = adaled.load('config.pt')
    mr = config.micro.make_multiresolution()
    loss = LayerLoss(config, mr)

    cmap = plt.cm.get_cmap('RdBu')
    mask_color = mpl.colors.to_rgba_array('yellow')[0, :3]

    args = parser.parse_args()
    alphas = [downscale(alpha.numpy()[None, :, :], args.downscale)[0, :, :, None]
              for alpha in loss.layer_loss_weights]

    with h5py.File(args.path, 'r') as f:
        group = f[args.group]
        for frame in args.frames:
            layers_channels = []
            for layer in range(len(alphas)):
                channels = group[f'layers/layer{layer}'][frame, 0, :, :, :]
                channels = downscale(channels, args.downscale)
                layers_channels.append(channels)

            for c in range(len(layers_channels[0])):
                scale = 0.0
                for channels in layers_channels:
                    scale = max(scale, np.quantile(np.abs(channels[c]), 0.99))
                print(f"frame={frame} c={c} scale={scale}")
                for layer, channels in enumerate(layers_channels):
                    prefix = f'{args.output_prefix}-L{layer}-F{frame:07d}-C{c}'
                    rgb = cmap((channels[c] + scale) / (2 * scale))[:, :, :3]

                    path = f'{prefix}.png'
                    print(f"Storing {path}.")
                    plt.imsave(path, rgb)

                    rgb_mask = (1 - alphas[layer]) * mask_color[None, None, :] + alphas[layer] * rgb
                    rgb_mask = np.clip(rgb_mask, 0.0, 1.0)
                    path = f'{prefix}-mask.png'
                    print(f"Storing {path}.")
                    plt.imsave(path, rgb_mask)


if __name__ == '__main__':
    main()
