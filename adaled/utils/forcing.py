from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.misc import get_global_default_np_dtype

import numpy as np

from typing import Callable, Optional, Sequence, Union

__all__ = ['ForcingConfig', 'ForcingFunc']


class ForcingFunc:
    """Forcing function with a precomputed profile, see `ForcingConfig`."""
    def __init__(self, F: np.ndarray, dt: float):
        if F.ndim <= 1:
            raise TypeError(f"F must contain the batch dimension, F.shape={F.shape}")
        self.F = F
        self.dt = dt

    def __call__(self, t: Union[float, np.ndarray]):
        k = np.asarray(t / self.dt, dtype=self.F.dtype)
        i = k.astype(np.int32)
        k -= i
        # The following formula produces a constant output for constant F.
        # k[..., None] in order to work with multiple `t`.
        return self.F[i] + (self.F[i + 1] - self.F[i]) * k[..., None]
        # return self.F[i] * (1 - k) + self.F[i + 1] * k

    def debug(self, path: str = 'F-debug.csv'):
        """Save the forcing function profile to the given path."""
        t = self.dt * np.arange(len(self.F))[:, None]
        np.savetxt(path, np.concatenate([t, self.F], axis=-1),
                   delimiter=',', header='t,F')
        print(f"Forcing func profile saved to {path!r}.")


@dataclass
class ForcingConfig(DataclassMixin):
    """Configurable profile of external forcing.

    Kinds:
        levels-cyclic: cycle between `T`-long levels of F listed in `values`
        levels-random: randomize levels in the range specified by `values`

    Arguments:
        decay_T: (float) time scale of the transition between different values
        brownian_amplitude: (float) gaussian noise added each time step to the
                current value of F
        smooth_T: (float) sigma of gaussian smoothing to apply, applied as a
                final step
    """
    values: Sequence[float]
    T: float
    kind: str = 'levels-cyclic'
    seed: int = 12345

    # Time to approach the target F by a factor of 1/e. Sudden by default.
    decay_T: float = 1e-6

    smooth_T: float = 0.0

    # Brownian noise to add each time step, defaults to 0.
    brownian: float = 0.0

    profile_output_path: Optional[str] = None

    # TODO: Add per-simulation seed somehow.
    def make_func(
            self,
            max_steps: int,
            dt: float,
            num_simulations: int = 1,
            dtype: Union[type, str] = 'default') -> ForcingFunc:
        """Prepare the profile for `max_steps` time steps of duration `dt`."""
        assert max_steps < 1e8, "max_steps too high? memory usage of F func " \
                                "currently is linear with respect to max_steps"
        if dtype == 'default':
            dtype = get_global_default_np_dtype()

        max_steps += 100  # Safety margin.

        t = dt * np.arange(max_steps)
        values = np.asarray(self.values)
        rng = np.random.default_rng(self.seed)
        if self.kind == 'levels-cyclic':
            k = np.asarray(t // self.T).astype(np.int32)
            target = values[k % len(values)]
        elif self.kind == 'levels-random':
            if len(values) != 2:
                raise ValueError("expected two values (min and max), got {values}")
            num_cycles = int(np.ceil(max_steps * dt / self.T))
            values = rng.uniform(*values, 2 + num_cycles)
            k = np.asarray(t // self.T).astype(np.int32)
            target = values[k % len(values)]
        else:
            raise ValueError(f"unrecognized kind: {self.kind}")

        u = rng.uniform(-self.brownian, self.brownian, max_steps)
        F = np.zeros(max_steps)
        curr = target[0]
        decay = np.exp(-dt / self.decay_T)
        for i in range(0, max_steps):
            F[i] = curr = target[i] + decay * (curr - target[i]) + u[i]

        if self.smooth_T:
            from scipy.ndimage import gaussian_filter1d
            F = gaussian_filter1d(F, self.smooth_T / dt)

        # Expand along simulation at axis 1.
        F = np.broadcast_to(F, (num_simulations,) + F.shape)
        F = np.moveaxis(F, 0, 1)
        F = F.astype(dtype)

        func = ForcingFunc(F, dtype(dt))
        if self.profile_output_path:
            func.debug(self.profile_output_path)
        return func
