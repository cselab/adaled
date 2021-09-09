import cubismup2d as cup2d  # Import immediately due to OpenMP versions.
import adaled
from ..common.postprocess import Postprocess, main

import numpy as np

from typing import List, Optional, Tuple

class CylinderPostprocess(Postprocess):
    def get_forces_integration_segments(self, lift_h: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.config.micro.compute_h()
        r = self.config.micro.r + lift_h * h  # Lifted surface.
        cx, cy = self.config.micro.center

        N = int(np.ceil(2.0 * 2 * np.pi * r / h))  # ~2 segments per cell.
        dtheta = 2 * np.pi / N
        theta = np.arange(N) * dtheta

        cos = np.cos(theta)
        sin = np.sin(theta)
        x = cx + r * cos
        y = cy + r * sin
        nx = cos
        ny = sin
        dl = np.full(N, r * dtheta)
        return np.stack([x, y], axis=1), np.stack([nx, ny], axis=1), dl


if __name__ == '__main__':
    main(CylinderPostprocess)
