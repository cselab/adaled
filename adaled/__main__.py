from typing import List, Optional
import adaled
import argparse
import sys

def describe(path: str):
    """Load and print the metadata of a given file."""
    if path.endswith('.pt'):
        data = adaled.load(path)
        if isinstance(data, adaled.TensorCollection):
            data.describe()
        else:
            print(data)
    elif path.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            f.visititems(print)
    else:
        raise NotImplementedError(f"extension not supported: {path}")


def main(argv: Optional[List[str]]):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="command", dest='command', help="sub-command help")
    add = subparsers.add_parser('describe', help="describe given files, print metadata").add_argument
    add('files', type=str, nargs='+', help="paths of files to describe")

    args = parser.parse_args(argv)
    if args.command == 'describe':
        for path in args.files:
            describe(path)
    else:
        parser.print_help()


if __name__ == '__main__':
    main(sys.argv[1:])
