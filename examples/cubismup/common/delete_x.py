#!/usr/bin/env python3

"""Script for deleting large x_micro and x_macro datasets from record HDF5 files.

The script is not really fully automized (because deleting stuff from HDF5 is
not automated in general). Follow the instructions printed by the script.

Usage:
    python3 -m examples.cubismup.common.delete_x [--doit] record-0*-0*.h5

The weird file pattern is to avoid the record-000-latest.h5 symbol link.
Without --doit, a dry run is performed. First make sure that all files were
selected correctly before running the script.

Try not to break the script (the real run) once it starts, it will leave a
bunch of files renamed, without preparing next stages.
"""

import argparse
import os
import re
import subprocess

import h5py

from typing import List, Optional

def main(argv: Optional[List[str]]):
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--doit', action='store_true', default=False, help="Actually delete x_micro and x_macro")
    add('paths', nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    out_script = ['#!/bin/bash\n\nset -ex\n']

    for path in args.paths:
        print(f"Opening {path}")

        with h5py.File(path, 'r+' if args.doit else 'r') as f:
            pattern = re.compile(r'fields/simulations/x/m(a|i)cro/layers')

            to_delete = []
            def func(name, d):
                if isinstance(d, h5py.Group) and pattern.match(name):
                    to_delete.append(name)

            f.visititems(func)

            if args.doit:
                for name in to_delete:
                    print(f"Deleting {name}")
                    del f[name]
                print("Renaming...")
                tmp_path = os.path.join(os.path.dirname(path), '__' + os.path.basename(path))
                os.rename(path, tmp_path)
                out_script.append(f'h5repack {tmp_path} {path}')
            else:
                for name in to_delete:
                    print(f"Would delete: {name}")

    if args.doit:
        with open('__delete_x_finalize.sh', 'w') as f:
            out_script.append('')
            f.write('\n'.join(out_script))
        print("\n\nRun:\nbash __delete_x_finalize.sh")
        print("and then delete __*.h5 files")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
