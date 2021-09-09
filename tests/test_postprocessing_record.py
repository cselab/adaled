from base import TestCase
from adaled.postprocessing.record import get_cycle_slices
import adaled

import numpy as np

class TestPostprocessingStatistics(TestCase):
    def test_get_cycle_slices(self):
        Stages = adaled.AdaLEDStage
        WARMUP = np.int32(Stages.WARMUP)
        CMP = np.int32(Stages.COMPARISON)
        MACRO = np.int32(Stages.MACRO)
        MICRO = np.int32(Stages.MICRO)
        RELAX = np.int32(Stages.RELAXATION)

        stages = np.array([
            WARMUP, CMP, MACRO, RELAX,
            WARMUP, CMP, MACRO, RELAX,
            WARMUP, WARMUP, CMP, CMP, MICRO, MICRO, MICRO,
            CMP, MACRO,  # Test no macro and no relax.
            CMP, MACRO,  # Test no macro and no relax.
        ])
        expected = [
            [0, 4],
            [4, 8],
            [8, 15],
            [15, 17],
            [17, 19],
        ]
        assert expected[-1][1] == len(stages)

        self.assertArrayEqual(get_cycle_slices(stages), expected)
