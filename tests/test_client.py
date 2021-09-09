from base import TestCase
from adaled.led import AdaLEDStage as Stage, AdaLEDStep as Step
import adaled

import numpy as np

from typing import List, Tuple
import os

class _DummyMicro(adaled.MicroSolver):
    def __init__(self):
        self.state = np.zeros((1, 3))

    def advance(self, F: np.ndarray):
        self.state = self.state + 1000
        return self.get_state()

    def get_state(self):
        return self.state

    def update_state(self, new_state, skip_steps):
        self.state = np.array(new_state)  # Copy.


class _DummyMacro(adaled.MacroSolver):
    def __init__(self):
        super().__init__()
        self.unc = None
        self.state = None

    def advance(self, batch, hidden=None, *,
                compute_uncertainty: bool = False,
                teacher_forcing: bool = False) -> 'arraylike':
        if teacher_forcing:
            self.unc = np.array([0.15])
            self.z = batch['z'] + 1001.0
        else:
            self.unc = self.unc + 0.10
            self.z = self.z + 1002.0

        # Always return uncertainty.
        return (self.z, None, self.unc)


class TestClient(TestCase):
    def _assertSteps(self, computed: List[Step], expected: List[Tuple]):
        for step, (stage, x, z, unc) in zip(computed, expected):
            # print(step, stage, x, z, unc)
            unc = np.float64(unc) if unc is not None else unc
            self.assertEqual(step.stage, stage)
            self.assertEqual(step.x is None, x is None)
            self.assertEqual(step.z is None, z is None)
            if step.x is not None:
                self.assertArrayEqual(step.x, np.float64(x))
            if step.z is not None:
                self.assertArrayEqual(step.z, np.float64(z))
            if step.uncertainty is None or unc is None:
                self.assertEqual(step.uncertainty, unc)
            else:
                self.assertArrayAlmostEqual(step.uncertainty, unc)
        self.assertEqual(len(computed), len(expected))

    def test_partially_accepted_cycle(self):
        micro = _DummyMicro()
        macro = _DummyMacro()
        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=3,
                k_cmp=5,
                max_cmp_error=10000.0,
                max_uncertainty=0.80,
                max_macro_steps=1000,
                max_micro_steps=1000,
                num_relaxation_steps=3).create()

        config = adaled.AdaLEDClientConfig(always_run_micro=True, quiet=True)
        client = adaled.AdaLEDClient(config, macro, criteria)
        generator = client.make_generator(micro)

        computed_steps = list(generator.start_cycle())
        expected_steps = [
            (Stage.WARMUP,      1000.,  1001., 0.15),
            (Stage.WARMUP,      2000.,  2001., 0.15),
            (Stage.WARMUP,      3000.,  3001., 0.15),
            (Stage.COMPARISON,  4000.,  4001., 0.15),
            (Stage.COMPARISON,  5000.,  5003., 0.25),
            (Stage.COMPARISON,  6000.,  6005., 0.35),
            (Stage.COMPARISON,  7000.,  7007., 0.45),
            (Stage.COMPARISON,  8000.,  8009., 0.55),
            (Stage.MACRO,       9000.,  9011., 0.65),
            (Stage.MACRO,      10000., 10013., 0.75),
            (Stage.RELAXATION, 11013.,  None, None),  # 0.85
            (Stage.RELAXATION, 12013.,  None, None),
            (Stage.RELAXATION, 13013.,  None, None),
        ]
        self._assertSteps(computed_steps, expected_steps)
        self.assertEqual(generator._cycle_diagnostics.validation_error, (10013 - 10000) ** 2)

    def test_fully_accepted_cycle(self):
        micro = _DummyMicro()
        macro = _DummyMacro()
        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=2,
                k_cmp=4,
                max_cmp_error=100000.0,
                max_uncertainty=3.80,
                max_macro_steps=5,
                max_micro_steps=1000,
                num_relaxation_steps=3).create()

        config = adaled.AdaLEDClientConfig(always_run_micro=True, quiet=True)
        client = adaled.AdaLEDClient(config, macro, criteria)
        generator = client.make_generator(micro)

        computed_steps = list(generator.start_cycle())
        expected_steps = [
            (Stage.WARMUP,      1000.,  1001., 0.15),
            (Stage.WARMUP,      2000.,  2001., 0.15),
            (Stage.COMPARISON,  3000.,  3001., 0.15),
            (Stage.COMPARISON,  4000.,  4003., 0.25),
            (Stage.COMPARISON,  5000.,  5005., 0.35),
            (Stage.COMPARISON,  6000.,  6007., 0.45),
            (Stage.MACRO,       7000.,  7009., 0.55),
            (Stage.MACRO,       8000.,  8011., 0.65),
            (Stage.MACRO,       9000.,  9013., 0.75),
            (Stage.MACRO,      10000., 10015., 0.85),
            (Stage.MACRO,      11000., 11017., 0.95),
            (Stage.RELAXATION, 12017., None, None),  # 1.05
            (Stage.RELAXATION, 13017., None, None),
            (Stage.RELAXATION, 14017., None, None),
        ]
        self._assertSteps(computed_steps, expected_steps)
        self.assertEqual(generator._cycle_diagnostics.validation_error, (11017 - 11000) ** 2)

    def test_rejected_cycle(self):
        micro = _DummyMicro()
        macro = _DummyMacro()
        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=2,
                k_cmp=4,
                max_cmp_error=1e-100,
                max_uncertainty=3.80,
                max_macro_steps=1000,
                max_micro_steps=5,
                num_relaxation_steps=3).create()

        config = adaled.AdaLEDClientConfig(always_run_micro=True, quiet=True)
        client = adaled.AdaLEDClient(config, macro, criteria)
        generator = client.make_generator(micro)

        computed_steps = list(generator.start_cycle())
        expected_steps = [
            (Stage.WARMUP,      1000., 1001., 0.15),
            (Stage.WARMUP,      2000., 2001., 0.15),
            (Stage.COMPARISON,  3000., 3001., 0.15),
            (Stage.COMPARISON,  4000., 4003., 0.25),
            (Stage.COMPARISON,  5000., 5005., 0.35),
            (Stage.COMPARISON,  6000., 6007., 0.45),
            (Stage.MICRO,       7000., None, None),
            (Stage.MICRO,       8000., None, None),
            (Stage.MICRO,       9000., None, None),
            (Stage.MICRO,      10000., None, None),
            (Stage.MICRO,      11000., None, None),
        ]
        self._assertSteps(computed_steps, expected_steps)
