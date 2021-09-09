#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main(argv=None):
    from adaled.led.main import parse_and_process_argv
    import examples.vdp.setup as setup

    config = setup.Config()
    parse_and_process_argv(argv, config=config)
    setup.main(config)


if __name__ == '__main__':
    main()
