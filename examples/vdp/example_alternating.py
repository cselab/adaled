#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from examples.vdp.run import main

if __name__ == '__main__':
    main(['--config-json', '{"mu.values": [3.0, 1.0]}'])
