from base import TestCase
from adaled import TensorCollection
from adaled.transformers.autoencoders import CompoundAutoencoder
from adaled.transformers.scaling import Scaling, ScalingTransformer
import adaled
import adaled.nn.conv as conv

import torch

class TestAutoencoders(TestCase):
    def test_compound_autoencoder(self):
        # Use a real transformer to test the order of the batch and the latent
        # dimension in transformer input/output.

        config = conv.ConvMLPEncoderConfig(
                input_shape=(3, 32, 64),  # x=64, y=32
                conv_layers_kernel_sizes=[2, 2],
                conv_layers_channels=[5, 6],
                conv_layers_strides=[2, 2],
                latent_state_dim=7,
                pool_kernel_sizes=[2, 2],
                pool_kernel_strides=[2, 2],
                pool_type='avg',
                padding_mode='circular',
                activation='celu',
                activation_output='tanh')

        encoder = adaled.ConvMLPEncoder(config)
        decoder = adaled.ConvMLPDecoder(encoder.make_decoder_config())
        ae_x = adaled.AutoencoderTransformer(encoder, decoder)
        ae_F = ScalingTransformer(Scaling(scale=4.0, shift=0.0))  # Use a nice factor.

        mapping = {
            'xx': (ae_x, 7),
            'FF': (ae_F, 1),
        }
        ae = CompoundAutoencoder(mapping)

        # Test x -> z transformation.
        x = torch.rand(10, 3, 32, 64)  # Batch size of 10.
        F = torch.rand(10, 1)
        batch_orig = adaled.TensorCollection(xx=x, FF=F)
        with torch.no_grad():
            batch_z = ae.transform(batch_orig)
        self.assertEqual(batch_z.shape, (10, 7 + 1))
        self.assertArrayEqual(batch_z[:, -1:], 4.0 * F, check_shape=True)

        # Test inverse transformation.
        with torch.no_grad():
            batch_x = ae.inverse_transform(batch_z)
        self.assertIsInstance(batch_x, adaled.TensorCollection)
        self.assertEqual(list(batch_x.keys()), ['xx', 'FF'])
        self.assertEqual(batch_x['xx'].shape, (10, 3, 32, 64))
        self.assertArrayEqual(batch_x['FF'], F, check_shape=True)

    def test_hierarchical_compound_autoencoder(self):
        scaling1 = ScalingTransformer(Scaling(scale=2.0, shift=0.0))
        scaling2 = ScalingTransformer(Scaling(scale=4.0, shift=0.0))
        scaling3 = ScalingTransformer(Scaling(scale=8.0, shift=0.0))
        scaling4 = ScalingTransformer(Scaling(scale=16.0, shift=0.0))

        hierarchy = {
            'AA': {
                'aa1': (scaling1, 2),
                'aa2': (scaling2, 3),
            },
            'BB': {
                'bb1': (scaling3, 4),
                'bb2': (scaling4, 5),
            },
        }
        ae = adaled.CompoundAutoencoder(hierarchy)
        x = TensorCollection({
            'AA': {
                'aa1': torch.tensor([[10., 20.]]),  # Batch size of 1.
                'aa2': torch.tensor([[10., 20., 30.]]),
            },
            'BB': {
                'bb1': torch.tensor([[10., 20., 30., 40.]]),
                'bb2': torch.tensor([[10., 20., 30., 40., 50.]]),
            },
        })
        z = ae.transform(x)
        self.assertArrayEqual(z, [[
            20., 40.,
            40., 80, 120.,
            80., 160., 240., 320.,
            160., 320., 480., 640., 800.,
        ]])
        x_back = ae.inverse_transform(z)
        self.assertCollectionEqual(x_back, x)

        # Test __getitem__.
        z = ae['AA'].transform(x['AA'])
        self.assertArrayEqual(z, [[20., 40., 40., 80., 120.]])
        self.assertCollectionEqual(ae['AA'].inverse_transform(z), x['AA'])

        z = ae['AA']['aa1'].transform(x['AA', 'aa1'])
        self.assertArrayEqual(z, [[20., 40.]])
        self.assertArrayEqual(ae['AA']['aa1'].inverse_transform(z), x['AA', 'aa1'])
