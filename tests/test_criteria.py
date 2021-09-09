from base import TestCase
from adaled import SimpleCriteria, SimpleCriteriaConfig

import numpy as np

class TestCriteria(TestCase):
    def test_random_duration(self):
        x = None
        F = None

        def _simulate(low, high, expected_low, expected_high, N):
            config = SimpleCriteriaConfig(max_micro_steps=(low, high))
            criteria = SimpleCriteria(config, seed=12345)
            counts = [0] * len(expected_low)
            for i in range(N):
                step = 0
                while criteria.should_continue_micro(step, x, F):
                    step += 1
                counts[step] += 1

            if not all(elo <= c <= ehi
                       for c, elo, ehi in zip(counts, expected_low, expected_high)):
                self.fail(f"low={expected_low}  counts={counts}  high={expected_high}")


        _simulate(0, 0, [1000, 0, 0], [1000, 0, 0], 1000)
        _simulate(0, 1, [450, 450, 0], [550, 550, 0], 1000)
        _simulate(0, 2, [3000, 3000, 3000, 0], [3600, 3600, 3600, 0], 10000)
        _simulate(2, 4, [0, 0, 3000, 3000, 3000, 0], [0, 0, 3600, 3600, 3600, 0], 10000)
        _simulate(2, 5, [0, 0, 2300, 2300, 2300, 2300, 0],
                  [0, 0, 2700, 2700, 2700, 2700, 0], 10000)
