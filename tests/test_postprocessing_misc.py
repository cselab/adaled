from base import TestCase
from adaled.postprocessing.misc import pareto_front

import numpy as np

class TestPostprocessingMisc(TestCase):
    def test_pareto_front_mask_small(self):
        # Oppositely sorted arrays, all points are on the Pareto front.
        x = [10.0, 20.0, 50.0, 100.0]
        y = [100.0, 50.0, 20.0, 10.0]
        self.assertArrayEqual(pareto_front(x, y), [0, 1, 2, 3])
        self.assertArrayEqual(pareto_front(y, x), [3, 2, 1, 0])  # Reverse.

        # Sorted array, only the last point is on the Pareto front.
        x = [10.0, 20.0, 50.0, 100.0]
        y = [10.0, 20.0, 50.0, 100.0]
        self.assertArrayEqual(pareto_front(x, y), [3])
        self.assertArrayEqual(pareto_front(x[::-1], y[::-1]), [0])  # Reverse.

        # More complicated example.
        x = [10, 20, 15, 25, 30]
        y = [20, 5, 30, 15, 10]
        self.assertArrayEqual(pareto_front(x, y), [2, 3, 4])

        # Test multiple points with same x.
        x = [10, 10, 10, 10, 10]
        y = [30, 10, 20, 50, 40]
        self.assertArrayEqual(pareto_front(x, y), [3])
        self.assertArrayEqual(pareto_front(y, x), [3])

    def test_pareto_front_mask_random(self):
        random = np.random.RandomState(seed=11238)
        N = 100
        for t in range(30):
            xs = np.sort(random.uniform(0.0, 1.0, N))
            ys = random.uniform(0.0, 1.0, N)
            if len(set(xs)) != N or len(set(ys)) != N:
                continue  # Ignore if xs and ys are not unique.

            expected = []
            for i, (x, y) in enumerate(zip(xs, ys)):
                if not np.any((xs > x) & (ys > y)):
                    expected.append(i)
            self.assertArrayEqual(pareto_front(xs, ys), expected)
