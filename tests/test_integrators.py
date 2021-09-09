import numpy as np

from base import TestCase
import adaled
from adaled.utils.buffer import DynamicArray
from adaled.solvers.integrators import ForwardEulerMicroSolver


def ode_vdp_oscillator(t, X, mu):
    """Van der Pol oscillator:

    dx/dt = mu * (x - 1/3 x^3 - y)
    dy/dt = 1/mu * x

    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    """
    x = X[..., 0]
    y = X[..., 1]
    return np.stack([
        mu * (x - (1. / 3) * x * x * x - y),
        x / mu,
    ], axis=-1)


class TestIntegrators(TestCase):
    def test_forward_euler(self):
        """
        Test the initial jump at t=0 with (x0, y0)=(0, 0).
        """
        mu = 1 / 0.001
        dt = 0.00005
        x = np.array([[1.0, -1.0]])

        solver = ForwardEulerMicroSolver(
                ode_vdp_oscillator, dt=dt, integrator_dt=dt, t=0.0, x=x)

        F = np.full((1,), mu)
        data = DynamicArray()
        for i in range(100):
            data.append(x[0])
            x = solver.advance(F)

        self.assertRegressionCSV(
                'vdp_forward_euler.csv', data,
                header='x,y', save=False)
