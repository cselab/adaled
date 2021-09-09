#!/usr/bin/env python3

import adaled

import numpy as np

from typing import Optional, Sequence
import argparse
import os

def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('dirs', type=str, nargs='+', help="Runs to compare.")
    args = parser.parse_args(argv)

    for dir in args.dirs:
        path = os.path.join(dir, 'postprocessed.pt')
        try:
            post = adaled.load(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping")

        is_macro = post['metadata', 'stage'] == adaled.AdaLEDStage.MACRO

        mse_abs: np.ndarray = post['errors', 'mse_abs']
        mse_rel: np.ndarray = post['errors', 'mse_rel']
        print("{}  is_macro: {:.2f}%  all avg: {:.5f} {:.5f}  only macro: {:.5f} {:.5f}".format(
                dir,
                100 * is_macro.mean(),
                mse_abs.mean(),
                mse_rel.mean(),
                mse_abs[is_macro].mean(),
                mse_rel[is_macro].mean()))


if __name__ == '__main__':
    main()
