from adaled.nn.loss import weighted_mse_losses, weighted_mean_l1_losses
from base import TestCase

import torch

class TestNNLosses(TestCase):
    def setUp(self):
        self.expected = torch.tensor([
            [0., -1., 0.,  10.,  0., 0.],
            [1., -2., 1., -11., -1., 1.],
        ], requires_grad=True)
        self.computed = self.expected + torch.tensor([
            [1., -1., 0., 0., 3., -3.],
            [0., 0., 2., -2., 0., 0.],
        ], requires_grad=True)
        self.w = torch.tensor([
            [0., 0., 0., 5., 10., 20.],
            [0., 0., 0., 30., 0., 40.],
        ], requires_grad=True)

    def _test(self, loss_func, w0r0, w0r1, w1r0, w1r1):
        def test(expected, **kw):
            computed = loss_func(self.computed, self.expected, **kw)
            self.assertArrayAlmostEqual(computed.detach(), expected)

        def test_all():
            test(w0r0, weight=None, relative=False)
            test(w0r1, weight=None, relative=True, eps=0.0)
            test(w1r0, weight=self.w, relative=False)
            test(w1r1, weight=self.w, relative=True, eps=0.0)

        with torch.no_grad():
            test_all()  # Test first without grad, then with grad.
        test_all()

    def test_weighted_mse_losses(self):
        w0r0 = [(1 + 1 + 9 + 9) / 6, (4 + 4) / 6]
        w0r1 = [(1 + 1 + 9 + 9) / (1 + 100), (4 + 4) / (1 + 4 + 1 + 121 + 1 + 1)]
        w1r0 = [(10 * 9 + 20 * 9) / 6, (30 * 4) / 6]
        w1r1 = [(10 * 9 + 20 * 9) / (5 * 100), (30 * 4) / (30. * 121 + 40. * 1)]
        self._test(weighted_mse_losses, w0r0, w0r1, w1r0, w1r1)

    def test_weighted_mean_l1_losses(self):
        w0r0 = [(1 + 1 + 3 + 3) / 6, (2 + 2) / 6]
        w0r1 = [(1 + 1 + 3 + 3) / (1 + 10), (2 + 2) / (1 + 2 + 1 + 11 + 1 + 1)]
        w1r0 = [(10 * 3 + 20 * 3) / 6, (30 * 2) / 6]
        w1r1 = [(10 * 3 + 20 * 3) / (5 * 10), (30 * 2) / (30 * 11 + 40 * 1)]
        self._test(weighted_mean_l1_losses, w0r0, w0r1, w1r0, w1r1)
