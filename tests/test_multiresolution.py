from base import TestCase
from adaled.transformers.multiresolution import \
        DEFAULT_UPSCALING_KWARGS, MRLayerConfig, Multiresolution, \
        FinalizedLayer, sigmoid

import numpy as np

class TestMultiresolution(TestCase):
    def assertLayerEqual(self, layer: FinalizedLayer, **kwargs):
        for key, value in kwargs.items():
            self.assertEqual(getattr(layer, key), value, key)

    def test_simple_construction(self):
        config = [
            MRLayerConfig(stride=2),
            MRLayerConfig(center=(0.50, 0.25), size=(0.20, 0.20)),
        ]
        mr = Multiresolution(config, grid=(1000, 500))
        self.assertLayerEqual(mr.layers[0], begin=(0, 0), end=(1000, 500), stride=(2, 2))
        self.assertLayerEqual(mr.layers[1], begin=(400, 150), end=(600, 350), stride=(1, 1))

    def test_rounding(self):
        config = [
            MRLayerConfig(stride=2),
            MRLayerConfig(center=(0.50, 0.25), size=(0.20, 0.20), round_size_to=120),
        ]
        mr = Multiresolution(config, grid=(1000, 500))
        self.assertLayerEqual(mr.layers[1], begin=(380, 130), end=(620, 370), stride=(1, 1))

    def test_downscaling_small(self):
        mr = Multiresolution([MRLayerConfig(stride=2)], grid=(6, 4))
        x = np.array([
            [0., 1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5., 6.],
            [2., 3., 4., 5., 6., 7.],
            [3., 4., 5., 6., 7., 8.],
        ])
        x0 = np.array([
            [1., 3., 5.],
            [3., 5., 7.],
        ])
        self.assertArrayAlmostEqual(mr.slice_and_downscale(x)[0], x0, check_backend=True)

    def test_downscaling_large(self):
        config = [
            MRLayerConfig(stride=2),
            MRLayerConfig(center=(400, 200), size=(300, 200), stride=1),
        ]
        mr = Multiresolution(config, grid=(1000, 500))
        x = np.random.uniform(0.0, 1.0, (500, 1000))

        x0, x1 = mr.slice_and_downscale(x)
        self.assertArrayAlmostEqual(x0, x.reshape(250, 2, 500, 2).mean((-1, -3)), check_backend=True)
        self.assertArrayAlmostEqual(x1, x[100:300, 250:550], check_backend=True)

    def test_upscaling_order0(self):
        kwargs = {'mode': 'nearest'}
        mr = Multiresolution([MRLayerConfig(stride=2, upscaling_kwargs=kwargs)], grid=(6, 4))
        x0 = np.array([
            [1., 3., 5.],
            [3., 5., 7.],
        ])
        x = np.array([
            [1., 1., 3., 3., 5., 5.],
            [1., 1., 3., 3., 5., 5.],
            [3., 3., 5., 5., 7., 7.],
            [3., 3., 5., 5., 7., 7.],
        ])
        self.assertArrayAlmostEqual(mr.layers[0].upscale(x0), x)

    def test_upscaling_order1(self):
        kwargs = {'mode': 'bilinear', 'align_corners': False}
        mr = Multiresolution([MRLayerConfig(stride=2, upscaling_kwargs=kwargs)], grid=(6, 4))

        # Try 1D (every row equal). Expecting cell-centered-like everywhere
        # except at the boundaries.
        x0 = np.array([
            [1., 3., 5.],
            [1., 3., 5.],
        ])
        x = np.array([
            [1.0, 1.5, 2.5, 3.5, 4.5, 5.],
            [1.0, 1.5, 2.5, 3.5, 4.5, 5.],
            [1.0, 1.5, 2.5, 3.5, 4.5, 5.],
            [1.0, 1.5, 2.5, 3.5, 4.5, 5.],
        ])
        self.assertArrayAlmostEqual(mr.layers[0].upscale(x0), x)

        # Try 2D. Results not verified whether they make sense, using as a
        # regression test.
        x0 = np.array([
            [1., 3., 5.],
            [3., 5., 7.],
        ])
        x = np.array([
            [1. , 1.5, 2.5, 3.5, 4.5, 5. ],
            [1.5, 2. , 3. , 4. , 5. , 5.5],
            [2.5, 3. , 4. , 5. , 6. , 6.5],
            [3. , 3.5, 4.5, 5.5, 6.5, 7. ],
        ])
        self.assertArrayAlmostEqual(mr.layers[0].upscale(x0), x)

    def test_rebuild_1d(self):
        upscaling = {'mode': 'nearest'}
        config = [
            MRLayerConfig(stride=4, upscaling_kwargs=upscaling),
            MRLayerConfig(center=(400,), size=(400,), stride=2,
                          upscaling_kwargs=upscaling,
                          alpha_margin_cells=22.0, alpha_sigma_cells=0.7),
            MRLayerConfig(center=(500,), size=(100,), stride=1,
                          alpha_margin_cells=21.0, alpha_sigma_cells=0.6),
        ]
        mr = Multiresolution(config, grid=(1000,), alpha_dtype=np.float64)

        self.assertEqual(mr.layers[0].box, ((0,), (1000,)))
        self.assertEqual(mr.layers[1].box, ((200,), (600,)))
        self.assertEqual(mr.layers[2].box, ((450,), (550,)))
        self.assertEqual(mr._relative_slices[0], (..., slice(200, 600)))
        self.assertEqual(mr._relative_slices[1], (..., slice(250, 350)))

        x = 0.5 + np.arange(1000)
        dist1 = np.minimum(x - (200 + 22.0), 600 - 22.0 - x) / 0.7
        dist2 = np.minimum(x - (450 + 21.0), 550 - 21.0 - x) / 0.6
        alpha1 = sigmoid(dist1)
        alpha2 = sigmoid(dist2)
        mr._compute_alphas()
        self.assertEqual(mr._alphas[0].shape, (400,))
        self.assertEqual(mr._alphas[1].shape, (100,))
        self.assertArrayAlmostEqual(mr._alphas[0], alpha1[200:600])
        self.assertArrayAlmostEqual(mr._alphas[1], alpha2[450:550])

        def solve():
            out = (1 - alpha1) * np.repeat(x0, 4)
            out[200:600] += (alpha1 - alpha2)[200:600] * np.repeat(x1, 2)
            out[450:550] += alpha2[450:550] * x2
            return out

        x0 = np.full((250,), 10.0)
        x1 = np.full((200,), 100.0)
        x2 = np.full((100,), 1000.0)
        expected = solve()
        computed = mr.rebuild_unsafe([x0, x1, x2])  # May modify x0, x1 and x2.
        self.assertArrayAlmostEqual(computed, expected)

        x0 = np.random.uniform(0.0, 1.0, 250)
        x1 = np.random.uniform(0.0, 10.0, 200)
        x2 = np.random.uniform(0.0, 100.0, 100)
        expected = solve()
        computed = mr.rebuild_unsafe([x0, x1, x2])
        self.assertArrayAlmostEqual(computed, expected)

    def test_rebuild_2d(self):
        upscaling = {'mode': 'nearest'}
        config = [
            MRLayerConfig(stride=4, upscaling_kwargs=upscaling),
            MRLayerConfig(center=(300, 200), size=(300, 160), stride=2,
                          upscaling_kwargs=upscaling,
                          alpha_margin_cells=12.0, alpha_sigma_cells=0.3),
            MRLayerConfig(center=(250, 200), size=(80, 80), stride=1,
                          alpha_margin_cells=11.0, alpha_sigma_cells=0.2),
        ]
        mr = Multiresolution(config, grid=(800, 400), alpha_dtype=np.float64)

        self.assertEqual(mr.layers[0].box, ((0, 0), (800, 400)))
        self.assertEqual(mr.layers[1].box, ((150, 120), (450, 280)))
        self.assertEqual(mr.layers[2].box, ((210, 160), (290, 240)))
        self.assertEqual(mr._relative_slices[0], (..., slice(120, 280), slice(150, 450)))
        self.assertEqual(mr._relative_slices[1], (..., slice(40, 120), slice(60, 140)))

        x = 0.5 + np.arange(800)
        y = 0.5 + np.arange(400)
        dist1x = np.minimum(x - (150 + 12.0), 450 - 12.0 - x) / 0.3
        dist1y = np.minimum(y - (120 + 12.0), 280 - 12.0 - y) / 0.3
        dist2x = np.minimum(x - (210 + 11.0), 290 - 11.0 - x) / 0.2
        dist2y = np.minimum(y - (160 + 11.0), 240 - 11.0 - y) / 0.2
        alpha1 = np.outer(sigmoid(dist1y), sigmoid(dist1x))
        alpha2 = np.outer(sigmoid(dist2y), sigmoid(dist2x))
        mr._compute_alphas()
        self.assertArrayAlmostEqual(mr._alphas[0], alpha1[120:280, 150:450])
        self.assertArrayAlmostEqual(mr._alphas[1], alpha2[160:240, 210:290])

        def repeat2d(x, r):
            x = np.repeat(x, r, axis=-2)
            x = np.repeat(x, r, axis=-1)
            return x

        def solve():
            out = (1 - alpha1) * repeat2d(x0, 4)
            out[..., 120:280, 150:450] += (alpha1 - alpha2)[120:280, 150:450] * repeat2d(x1, 2)
            out[..., 160:240, 210:290] += alpha2[160:240, 210:290] * x2
            return out

        x0 = np.full((100, 200), 10.0)
        x1 = np.full((80, 150), 100.0)
        x2 = np.full((80, 80), 1000.0)
        expected = solve()
        computed = mr.rebuild_unsafe([x0, x1, x2])  # May modify x0, x1 and x2.
        self.assertArrayAlmostEqual(computed, expected)

        x0 = np.random.uniform(0.0, 1.0, (100, 200))
        x1 = np.random.uniform(0.0, 10.0, (80, 150))
        x2 = np.random.uniform(0.0, 100.0, (80, 80))
        expected = solve()
        computed = mr.rebuild_unsafe([x0, x1, x2])
        self.assertArrayAlmostEqual(computed, expected)

        x0 = np.random.uniform(0.0, 1.0, (2, 3, 100, 200))
        x1 = np.random.uniform(0.0, 10.0, (2, 3, 80, 150))
        x2 = np.random.uniform(0.0, 100.0, (2, 3, 80, 80))
        expected = solve()
        computed = mr.rebuild_unsafe([x0, x1, x2])
        self.assertArrayAlmostEqual(computed, expected)

    def test_dimensionality(self):
        config = [MRLayerConfig(stride=2)]
        mr1 = Multiresolution(config, grid=(100,), alpha_dtype=np.float32)
        mr2 = Multiresolution(config, grid=(100, 120), alpha_dtype=np.float32)
        mr3 = Multiresolution(config, grid=(100, 120, 140), alpha_dtype=np.float32)

        x1 = np.random.uniform(0.0, 1.0, (50,))
        x2 = np.random.uniform(0.0, 1.0, (60, 50))
        x3 = np.random.uniform(0.0, 1.0, (70, 60, 50))
        self.assertEqual(mr1.rebuild_unsafe([x1]).shape, (100,))
        self.assertEqual(mr2.rebuild_unsafe([x2]).shape, (120, 100))
        self.assertEqual(mr3.rebuild_unsafe([x3]).shape, (140, 120, 100))

    def test_compute_downscaled_cell_centers(self):
        config = [
            MRLayerConfig(stride=4),
            MRLayerConfig(center=(350, 150), size=(300, 200), stride=2),
        ]
        mr = Multiresolution(config, grid=(800, 400), alpha_dtype=np.float64)

        p0 = mr.layers[0].compute_downscaled_cell_centers()
        p1 = mr.layers[1].compute_downscaled_cell_centers()

        x0 = 0.5 + np.arange(0, 800, 4)
        y0 = 0.5 + np.arange(0, 400, 4)
        x1 = 0.5 + np.arange(200, 500, 2)
        y1 = 0.5 + np.arange(50, 250, 2)
        self.assertEqual(p0.shape, (100, 200, 2))
        self.assertEqual(p1.shape, (100, 150, 2))
        self.assertArrayEqual(p0, np.stack(np.meshgrid(x0, y0), axis=-1))
        self.assertArrayEqual(p1, np.stack(np.meshgrid(x1, y1), axis=-1))
