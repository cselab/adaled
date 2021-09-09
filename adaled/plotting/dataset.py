#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import adaled.utils.io_ as io_

from typing import List, Optional
import argparse
import pickle

def postprocess_dataset(path):
    dataset = io_.load(path)
    states = dataset.as_states('trajectory')
    print("Dataset: ", states.shape)

    x = states['x']
    print("Dataset state statistics:")
    print("    mean: ", x.mean(axis=0))
    print("    std:  ", x.std(axis=0))


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('input_path', type=str, help="path of the dataset file")

    args = parser.parse_args(argv)

    postprocess_dataset(args.input_path)


if __name__ == '__main__':
    main()
