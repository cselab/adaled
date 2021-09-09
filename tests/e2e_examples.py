#!/usr/bin/env python3

from base import TestCase
import unittest

import os

DIR = os.path.dirname(os.path.abspath(__file__))

class ExamplesQuickTest(TestCase):
    """Run examples for several hundreds of steps (at least one cycle), to
    check if they are compatible with the current adaled interface.

    Later checking the results might be added.
    """
    def setUp(self):
        self._old_cwd = os.getcwd()
        cwd = os.path.join(DIR, 'output')
        os.makedirs(cwd, exist_ok=True)
        os.chdir(cwd)

    def tearDown(self):
        os.chdir(self._old_cwd)

    def test_vdp(self):
        from examples.vdp.run import main
        main(['--config-json', '{"led.max_steps": 500}'])


if __name__ == '__main__':
    unittest.main()
