#!/usr/bin/env python3

import adaled.utils.io_ as io_

import torch

from typing import List
import argparse
import glob


def get_named_tensors(network, prefix: str = '') -> List[torch.nn.Parameter]:
    if hasattr(network, 'named_buffers'):
        for k, v in network.named_buffers(prefix):
            if not k.endswith('.num_batches_tracked'):
                yield (k, v)
        yield from network.named_parameters(prefix)
    elif hasattr(network, 'model'):
        yield from get_named_tensors(network.model, prefix + 'model.')
    elif hasattr(network, 'propagators'):
        for i, prop in enumerate(network.propagators):
            yield from get_named_tensors(prop, f'{prefix}prop{i}.')
    else:
        raise NotImplementedError(network.__class__)


def flatten_network_params(network):
    """Return a flattened tensor of all network parameters."""
    params = _get_named_params(network)
    return torch.cat([params.ravel() for params in params])


def print_L2_weight_difference(networks):
    """Print the L2 difference in weights between consecutive dumps of the network.

    Useful to detect if the networks were trained at all or at which stage the
    training started to make no significant difference.
    """

    for i, (a, b) in enumerate(zip(networks[:-1], networks[1:])):
        a = dict(get_named_tensors(a))
        b = dict(get_named_tensors(b))
        assert list(a.keys()) == list(b.keys())
        l2_parts = {key: ((a[key] - b[key]) ** 2).sum() for key in a.keys()}
        l2_all = sum(l2_parts.values())
        print(f"L2(#{i}, #{i+1}) = {l2_all}")
        for key, l2 in l2_parts.items():
            if l2 > 0.03 * l2_all:
                print(f"    {key}  L2 = {l2}")
                print(f"    #{i}:", a[key].tolist())
                print(f"    #{i+1}:", b[key].tolist())


def postprocess_networks(paths: List[str]):
    """Postprocess networks of the same type (i.e. autoencoder or RNNs)."""
    paths = sorted(paths)

    networks = []
    for i, path in enumerate(paths):
        print(f"Loading #{i}/{len(paths)}: {path}")
        networks.append(io_.load(path))
    print_L2_weight_difference(networks)


def main(argv=None):
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('paths', type=str, nargs='+', help="list of networks to analyze")

    args = parser.parse_args(argv)

    postprocess_networks(args.paths)


if __name__ == '__main__':
    main()
