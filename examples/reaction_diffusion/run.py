#!/usr/bin/env python3

"""Make sure the example is loaded as examples.<example_name>.<file>, rather
than just <file>."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main(argv=None):
    import examples.reaction_diffusion.setup as setup

    setup.main(argv)


if __name__ == '__main__':
    main()
