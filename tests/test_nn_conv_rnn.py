from base import TestCase
from adaled.backends import TensorCollection
from adaled.nn.conv_rnn import ConvRNN, ConvRNNCell, ConvRNNConfig

import torch

class TestConvRNNCell(TestCase):
    def test_2d_lstm(self):
        batch_size, input_channels, height, width = 32, 3, 16, 16
        hidden_channels = 5
        kernel_size = 5

        model = ConvRNNCell(input_channels, hidden_channels, kernel_size,
                            ndim=2, cell_type='lstm')

        state = None
        X = torch.rand(2, batch_size, input_channels, height, width)
        for x in X:
            output, state = model(x, state)

            shape = (batch_size, hidden_channels, height, width)
            self.assertEqual(output.shape, shape)
            self.assertEqual(len(state), 2)
            self.assertEqual(state[0].shape, shape)
            self.assertEqual(state[1].shape, shape)


class TestConvRNN(TestCase):
    def test_2d_lstm(self):
        def _test(input_, layers, F, residual):
            assert layers[-1] == input_
            config = ConvRNNConfig(
                    ndim=2, input_channels=input_ + (2 if F else 0),
                    layer_channels=layers, append_F=F, residual=residual)
            rnn = config.make()

            batch_size = 16
            z = torch.randn(batch_size, input_, 128, 96)
            F = torch.randn(2, batch_size, 2)

            z, h = rnn(TensorCollection(z=z, F=F[0]))
            z, h = rnn(TensorCollection(z=z, F=F[1]), h)
            self.assertEqual(z.shape, (16, 3, 128, 96))
            self.assertEqual(len(h), len(layers))
            self.assertEqual(len(h[0]), 2)
            for h_, l in zip(h, layers):
                self.assertEqual(h_[0].shape, (16, l, 128, 96))

        # _test(3, [5], True, False)
        # _test(3, [5], True, True)
        _test(3, [3], False, False)
        _test(3, [3], False, True)
        _test(3, [8, 16, 12, 3], False, True)

    def test_total_kernel_range(self):
        def _test(kernel_sizes, dilations, expected):
            config = ConvRNNConfig(
                    ndim=2, input_channels=2,
                    layer_channels=[10] * len(kernel_sizes),
                    kernel_sizes=kernel_sizes, dilations=dilations)
            total =  config.total_kernel_range()
            self.assertEqual(total, expected, config)

        _test([5], [1], expected=2)
        _test([7], [1], expected=3)
        _test([5, 5], [1, 1], expected=4)
        _test([3], [2], expected=2)
        _test([5], [2], expected=4)

        #             0 1 2 3 4 5 6 7 8 9 10
        # X . . X . . X . . X . . X          (k=5, d=3)
        #                     X . X . X      (k=3, d=2)
        #                         X X X X X  (k=5, d=1)
        _test([5, 3, 5], [3, 2, 1], expected=10)
