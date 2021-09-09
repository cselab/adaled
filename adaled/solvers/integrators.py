from adaled.backends import get_backend
from adaled.solvers.base import MicroSolver

import numpy as np

from typing import Any, Callable
import copy

scipy = None  # Lazy import scipy.

class ScipyODEIntegratorMicroSolver(MicroSolver):
    """Integrates a given ODE using scipy integrator.

    Any kwargs passed to the constructor are forwarded to the
    `scipy.integrate.solve_ivp` integrator.
    """
    def __init__(self,
                 rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 dt: float,
                 x: np.ndarray,
                 t: float = 0.0,
                 **kwargs):
        global scipy
        import scipy  # Lazy import scipy.
        import scipy.integrate
        self.x = copy.deepcopy(x)
        self.t = 0.0
        self.dt = dt
        self.rhs = rhs
        self.integrate_kwargs = kwargs

    def advance(self, F: Any):
        t0 = self.t
        t1 = self.t + self.dt
        out = get_backend(self.x).empty_like(self.x)
        for i, element in enumerate(self.x):
            # solve_ivp can integrate only one element at a time.
            result = scipy.integrate.solve_ivp(
                    lambda t, x: self.rhs(t, x, F),
                    (t0, t1), element, t_eval=[t1], **self.integrate_kwargs)
            out[i] = result.y[..., 0]
        self.x = out
        self.t = t1
        return out

    def get_state(self):
        return self.x

    def update_state(self, new_state, skip_steps: int = 1):
        self.t += skip_steps * self.dt
        self.x = copy.deepcopy(new_state)


class ForwardEulerMicroSolver(MicroSolver):
    """Integrates the given RHS function using forward-Euler integrator.

    The RHS is a function f(t, x, F) = dx/dt.
    """
    def __init__(self,
                 rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 x: np.ndarray,
                 dt: float,
                 integrator_dt: float = float('inf'),
                 t: float = 0.0):
        self.x = copy.deepcopy(x)
        self.t = t
        self.integrator_dt = integrator_dt
        self.dt = dt
        self.rhs = rhs

    def advance(self, F: Any):
        dt_left = self.dt
        while True:
            dt = min(self.integrator_dt, dt_left)
            if dt <= 0:
                break

            self.x += dt * self.rhs(self.t, self.x, F)
            self.t += dt
            dt_left -= dt

        return self.x

    def get_state(self):
        return self.x

    def update_state(self, new_state, skip_steps: int = 1):
        self.t += skip_steps * self.dt
        self.x = get_backend(self.x).cast_from(copy.deepcopy(new_state))
