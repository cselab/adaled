from base import TestCase
from adaled.backends import TensorCollection
from adaled.nn.networks import PaddingLayer, ViewLayer
import adaled
import adaled.nn.conv as conv

import numpy as np
import torch.nn

import pickle

class TestConvNetworks(TestCase):
    def _make_conv_1d_mlp_encoder_circular(self):
        # Dummy values for testing.
        config = conv.ConvMLPEncoderConfig(
                input_shape=(3, 512),
                conv_layers_kernel_sizes=[3, 5, 7],
                conv_layers_channels=[10, 20, 30],
                conv_layers_strides=[1, 2, 4],
                latent_state_dim=11,
                pool_kernel_sizes=[1, 2, 4],
                pool_kernel_strides=[1, 2, 4],
                pool_type='avg',
                padding_mode='circular',
                dropout_keep_prob=0.75,
                activation='celu',
                activation_output='tanh',
                batch_norm=True)
        network = conv.ConvMLPEncoder(config)
        return network

    def test_conv_1d_mlp_encoder_circular(self):
        network = self._make_conv_1d_mlp_encoder_circular()

        expected_shapes = [
            (3, 512),
            (3, 512),
            (10, 512),
            (10, 512),
            # (10, 513),
            # (10, 512),
            (10, 512),
            (10, 512),
            (20, 256),  # After conv with stride=2.
            (20, 256),
            # (20, 258),
            (20, 128),  # After pool with stride=2.
            (20, 128),
            (20, 128),
            (30, 32),   # After conv with stride=4.
            # (30, 35),
            (30, 8),    # After pool with stride=4.
            (30, 8),
            (240,),
            (11,),
            (11,),
        ]
        expected_layers = [
            adaled.RemoveExternalForcingLayer(),
            torch.nn.Conv1d(3, 10, kernel_size=3, stride=1,
                            padding=1, padding_mode='circular', bias=False),
            torch.nn.BatchNorm1d(10, affine=False),
            # PaddingLayer(pad=(0, 1), mode='circular'),
            # torch.nn.AvgPool1d(kernel_size=1, stride=1),
            torch.nn.CELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv1d(10, 20, kernel_size=5, stride=2,
                            padding=2, padding_mode='circular', bias=False),
            torch.nn.BatchNorm1d(20, affine=False),
            # PaddingLayer(pad=(1, 1), mode='circular'),
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            torch.nn.CELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv1d(20, 30, kernel_size=7, stride=4,
                            padding=3, padding_mode='circular'),
            # PaddingLayer(pad=(1, 2), mode='circular'),
            torch.nn.AvgPool1d(kernel_size=4, stride=4),
            torch.nn.CELU(),
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            torch.nn.Linear(in_features=30*8, out_features=11, bias=True),
            torch.nn.Tanh(),
        ]
        self.assertEqualModuleStructures(network.layers, expected_layers)

        layer_sizes = conv.get_layer_dimensions(network.layers, (3, 512))
        self.assertListEqual(layer_sizes, expected_shapes)

    def test_conv_1d_mlp_decoder_circular(self):
        encoder = self._make_conv_1d_mlp_encoder_circular()
        network = conv.ConvMLPDecoder(encoder.make_decoder_config())

        expected_shapes = [
            (11,),
            (240,),
            (30, 8),
            (30, 128),
            (20, 128),
            (20, 128),
            (20, 128),
            (20, 128),
            (20, 512),
            (10, 512),
            (10, 512),
            (10, 512),
            (10, 512),
            (3, 512),
            (3, 512),
        ]
        expected_layers = [
            torch.nn.Linear(11, 240, bias=True),
            ViewLayer(shape=(-1, 30, 8)),
            torch.nn.Upsample(scale_factor=16, mode='linear'),
            torch.nn.Conv1d(30, 20, kernel_size=7, padding=3, padding_mode='circular', bias=False),
            torch.nn.BatchNorm1d(20, affine=False),
            torch.nn.CELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Upsample(scale_factor=4.0, mode='linear'),
            torch.nn.Conv1d(20, 10, kernel_size=5, padding=2, padding_mode='circular', bias=False),
            torch.nn.BatchNorm1d(10, affine=False),
            torch.nn.CELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv1d(10, 3, kernel_size=3, padding=1, padding_mode='circular'),
            torch.nn.Tanh(),
        ]
        self.assertEqualModuleStructures(network.layers, expected_layers)

        layer_sizes = conv.get_layer_dimensions(network.layers, (11,))
        self.assertEqual(layer_sizes, expected_shapes)

    def _make_conv_2d_mlp_encoder_circular(self):
        # Dummy values for testing.
        config = conv.ConvMLPEncoderConfig(
                input_shape=(3, 256, 512),  # x=512, y=256
                conv_layers_kernel_sizes=[3, 5, 7],
                conv_layers_channels=[10, 20, 30],
                conv_layers_strides=[1, 2, 4],
                latent_state_dim=11,
                pool_kernel_sizes=[1, 2, 4],
                pool_kernel_strides=[1, 2, 4],
                pool_type='avg',
                padding_mode='circular',
                dropout_keep_prob=0.75,
                activation='celu',
                activation_output='tanh',
                batch_norm=True)
        network = conv.ConvMLPEncoder(config)
        return network

    def test_conv_2d_mlp_encoder_circular(self):
        network = self._make_conv_2d_mlp_encoder_circular()

        expected_shapes = [
            (3, 256, 512),
            (3, 256, 512),
            (10, 256, 512),
            (10, 256, 512),
            # (10, 257, 513),
            # (10, 256, 512),
            (10, 256, 512),
            (10, 256, 512),
            (20, 128, 256),  # After conv with stride=2.
            (20, 128, 256),
            # (20, 130, 258),
            (20, 64, 128),  # After pool with stride=2.
            (20, 64, 128),
            (20, 64, 128),
            (30, 16, 32),   # After conv with stride=4.
            # (30, 19, 35),
            (30, 4, 8),    # After pool with stride=4.
            (30, 4, 8),
            (30*4*8,),
            (11,),
            (11,),
        ]
        expected_layers = [
            adaled.RemoveExternalForcingLayer(),
            torch.nn.Conv2d(3, 10, kernel_size=3, stride=1,
                            padding=1, padding_mode='circular', bias=False),
            torch.nn.BatchNorm2d(10, affine=False),
            # PaddingLayer(pad=(0, 1, 0, 1), mode='circular'),
            # torch.nn.AvgPool2d(kernel_size=1, stride=1),
            torch.nn.CELU(),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(10, 20, kernel_size=5, stride=2,
                            padding=2, padding_mode='circular', bias=False),
            torch.nn.BatchNorm2d(20, affine=False),
            # PaddingLayer(pad=(1, 1, 1, 1), mode='circular'),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.CELU(),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(20, 30, kernel_size=7, stride=4,
                            padding=3, padding_mode='circular'),
            # PaddingLayer(pad=(1, 2, 1, 2), mode='circular'),
            torch.nn.AvgPool2d(kernel_size=4, stride=4),
            torch.nn.CELU(),
            torch.nn.Flatten(start_dim=-3, end_dim=-1),
            torch.nn.Linear(in_features=30*4*8, out_features=11, bias=True),
            torch.nn.Tanh(),
        ]
        self.assertEqualModuleStructures(network.layers, expected_layers)

        layer_sizes = conv.get_layer_dimensions(network.layers, (3, 256, 512))
        self.assertListEqual(layer_sizes, expected_shapes)

    def test_conv_2d_mlp_decoder_circular(self):
        encoder = self._make_conv_2d_mlp_encoder_circular()
        network = conv.ConvMLPDecoder(encoder.make_decoder_config())

        expected_shapes = [
            (11,),
            (30*4*8,),
            (30, 4, 8),
            (30, 64, 128),
            (20, 64, 128),
            (20, 64, 128),
            (20, 64, 128),
            (20, 64, 128),
            (20, 256, 512),
            (10, 256, 512),
            (10, 256, 512),
            (10, 256, 512),
            (10, 256, 512),
            (3, 256, 512),
            (3, 256, 512),
        ]
        expected_layers = [
            torch.nn.Linear(11, 30*4*8, bias=True),
            ViewLayer(shape=(-1, 30, 4, 8)),
            torch.nn.Upsample(scale_factor=16, mode='bilinear'),
            torch.nn.Conv2d(30, 20, kernel_size=7, padding=3, padding_mode='circular', bias=False),
            torch.nn.BatchNorm2d(20, affine=False),
            torch.nn.CELU(),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Upsample(scale_factor=4.0, mode='bilinear'),
            torch.nn.Conv2d(20, 10, kernel_size=5, padding=2, padding_mode='circular', bias=False),
            torch.nn.BatchNorm2d(10, affine=False),
            torch.nn.CELU(),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(10, 3, kernel_size=3, padding=1, padding_mode='circular'),
            torch.nn.Tanh(),
        ]
        self.assertEqualModuleStructures(network.layers, expected_layers)

        layer_sizes = conv.get_layer_dimensions(network.layers, (11,))
        self.assertEqual(layer_sizes, expected_shapes)
