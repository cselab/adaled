from base import TestCase
from adaled.backends import get_backend
from adaled.backends.backend_numpy import NumpyBackend
from adaled.backends.backend_torch import TorchBackend
import adaled.backends as backends

import numpy as np
import torch

import unittest

class TestBackendsMixin:
    backend: backends.Backend

    def test_array(self):
        x = self.backend.array([0, 1, 2])
        y = get_backend(x).array([0, 1, 2])
        self.assertArrayEqualStrict(x, y)

    def test_zeros_like(self):
        x = 1.0 * self.backend.array([0, 1, 2])
        self.assertArrayEqualStrict(self.backend.zeros(3), get_backend(x).zeros_like(x))

    def test_moveaxis(self):
        x = self.backend.zeros((3, 4))
        self.assertEqual(self.backend.moveaxis(x, 0, 1).shape, (4, 3))

    def test_empty_like__shape_kwarg(self):
        x = self.backend.zeros((3, 4))
        y = self.backend.empty_like(x, shape=(10,))
        self.assertEqual(y.shape, (10,))
        if hasattr(x, 'layout'):
            self.assertEqual(x.layout, y.layout)
        if hasattr(x, 'device'):
            self.assertEqual(x.device, y.device)


class TestBackendNumpy(TestBackendsMixin, TestCase):
    backend = NumpyBackend


class TestBackendTorchCPU(TestBackendsMixin, TestCase):
    backend = TorchBackend('cpu')


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBackendTorchCUDA(TestBackendsMixin, TestCase):
    backend = TorchBackend('cuda')


class TestSwitchingBackends(TestCase):
    def test_cast(self):
        x = np.arange(3)
        y = torch.arange(3).cpu()
        self.assertArrayEqualStrict(TorchBackend('cpu').cast_from(x), y)
        self.assertArrayEqualStrict(NumpyBackend.cast_from(y), x)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cast_from_cuda(self):
        x = torch.arange(3).cuda()
        self.assertArrayEqualStrict(NumpyBackend.cast_from(x), np.arange(3))
        self.assertArrayEqualStrict(TorchBackend('cpu').cast_from(x), torch.arange(3).cpu())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cast_to_cuda(self):
        x = np.arange(3)
        self.assertArrayEqualStrict(TorchBackend('cuda').cast_from(x), torch.arange(3).cuda())

        x = torch.arange(3).cpu()
        self.assertArrayEqualStrict(TorchBackend('cuda').cast_from(x), torch.arange(3).cuda())
