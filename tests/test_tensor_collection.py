from base import TestCase

from adaled.backends import TensorCollection, cmap
from adaled.backends.soa import TCShape
import numpy as np

class TestTCShape(TestCase):
    def test_str(self):
        a = TensorCollection(x=TensorCollection(y=np.arange(3)), z=np.arange(4))
        shape = a.shape
        self.assertEqual(str(shape), "TCShape({'x': {'y': (3,)}, 'z': (4,)})")
        self.assertEqual(str(shape), repr(shape))

    def test_shape(self):
        a = TensorCollection(xx=np.zeros((2, 3, 4)), yy=dict(zz=np.zeros((3, 5))))
        expected = {'xx': (2, 3, 4), 'yy': {'zz': (3, 5)}}
        expected0 = {'xx': (3, 4), 'yy': {'zz': (5,)}}
        self.assertEqual(a.shape, TCShape(expected))
        self.assertEqual(a.shape, expected)
        self.assertEqual(a.shape[1:], expected0)
        self.assertEqual(a.shape['xx'], (2, 3, 4))


class TestTensorCollection(TestCase):
    def test_str(self):
        a = TensorCollection(x=TensorCollection(y=np.arange(5)))
        self.assertEqual(str(a), "<TensorCollection of shape {'x': {'y': (5,)}}>")
        self.assertEqual(str(a), repr(a))

    def test_comparison(self):
        a = TensorCollection()
        b = TensorCollection()
        self.assertCollectionEqual(a, a)
        self.assertCollectionEqual(a, b)

        a = TensorCollection({'x': np.arange(5)})
        b = TensorCollection({'x': np.arange(5)})
        self.assertCollectionEqual(a, a)
        self.assertCollectionEqual(a, b)

        a = TensorCollection(x=np.arange(5))
        b = TensorCollection(x=np.arange(5))
        self.assertCollectionEqual(a, a)
        self.assertCollectionEqual(a, b)

    def test_getitem(self):
        a = TensorCollection({
            'x': np.arange(5),
            'y': np.arange(6),
        })
        self.assertArrayEqual(a['x'], np.arange(5))
        self.assertArrayEqual(a['y'], np.arange(6))
        self.assertCollectionEqual(a[3], {'x': 3, 'y': 3})
        self.assertEqual(a[3]['x'], 3)
        self.assertEqual(a['x'][3], 3)

    def test_getitem_newaxis_and_ellipsis(self):
        """np.newaxis should add an axes of size 1."""
        x = np.arange(6).reshape(2, 3)
        a = TensorCollection(x=x)
        self.assertCollectionEqual(a[np.newaxis], {'x': x[np.newaxis]})
        self.assertCollectionEqual(a[..., 0], {'x': x[..., 0]})

    def test_getitem_tuple(self):
        xy = np.arange(24).reshape(2, 3, 4)
        a = TensorCollection({'x': {'y': xy}})
        self.assertCollectionEqual(a['x'], {'y': xy})
        self.assertIs(a['x', 'y'], xy)
        self.assertArrayEqual(a['x', 1:2, 'y'], xy[1:2])
        self.assertArrayEqual(a[1:2, 'x', :, 'y', 3], xy[1:2, :, 3])

        with self.assertRaises(KeyError):
            a['abc']
        with self.assertRaises(KeyError):
            a['x', 'y', 'z']

    def test_getitem_array_index(self):
        x = 10 * np.arange(24)
        a = TensorCollection(x=x)

        idx = np.array([10, 15, 18])
        self.assertCollectionEqual(a[idx], {'x': x[idx]})

        mask = np.arange(24) % 3 == 0
        self.assertCollectionEqual(a[mask], {'x': x[mask]})

    def test_setitem_slice(self):
        a = TensorCollection({
            'x': np.arange(5),
            'y': np.arange(6),
        })
        a[1:3] = -1
        self.assertArrayEqual(a['x'], [0, -1, -1, 3, 4])
        self.assertArrayEqual(a['y'], [0, -1, -1, 3, 4, 5])

    def test_setitem_int(self):
        a = TensorCollection(xx=np.arange(4), yy={'zz': np.arange(3)})
        a[1] = -1
        self.assertCollectionEqual(a, {'xx': [0, -1, 2, 3], 'yy': {'zz': [0, -1, 2]}})

    def test_pop(self):
        x = np.arange(5)
        y = np.arange(5)
        a = TensorCollection(x=x, y=y)
        self.assertArrayEqualStrict(a.pop('y'), y)
        self.assertCollectionEqual(a, {'x': x})

        with self.assertRaises(KeyError):
            a.pop('i-do-not-exist')

    def test_backends_cmap_single_collection(self):
        aa = TensorCollection({'x': np.arange(5), 'y': np.arange(3)})
        out = cmap((lambda a: 3 * a), aa)
        self.assertCollectionEqual(out, {'x': [0, 3, 6, 9, 12], 'y': [0, 3, 6]})

    def test_backends_cmap_multiple_collections(self):
        aa = TensorCollection({'x': np.arange(5), 'y': np.arange(3)})
        bb = TensorCollection({'x': np.arange(5), 'y': np.arange(3)})
        cc = TensorCollection({'x': np.arange(5), 'y': np.arange(3)})

        out = cmap((lambda a, b, c: a + b + c), aa, bb, cc)
        self.assertCollectionEqual(out, {'x': [0, 3, 6, 9, 12], 'y': [0, 3, 6]})

    def test_backends_cmap_dict(self):
        out = cmap(lambda x: 2 * x, {'abc': 10})
        self.assertIsInstance(out, TensorCollection)
        self.assertCollectionEqual(out, {'abc': 20})

        out = cmap(lambda x, y: x + y, {'abc': 10}, {'abc': 20})
        self.assertIsInstance(out, TensorCollection)
        self.assertCollectionEqual(out, {'abc': 30})

    def test_allitems(self):
        y = z = np.arange(5)
        a = TensorCollection({'x': {'y': y}, 'z': z})
        self.assertEqual(list(a.allitems()), [(('x', 'y'), y), (('z',), z)])

    def test_allvalues(self):
        y = z = np.arange(5)
        a = TensorCollection({'x': {'y': y}, 'z': z})
        self.assertEqual(list(a.allvalues()), [y, z])

    def test_concat_flatten(self):
        x = np.arange(5)
        y = np.arange(6)
        aa = TensorCollection(x=x, y={'y': y})
        self.assertCollectionEqual(aa.concat_flatten(), {'x': x, 'y.y': y})

    def test_remove_empty(self):
        t = TensorCollection(x=np.arange(5), y={}, z={'w': {}})
        self.assertCollectionEqual(t.remove_empty(), {'x': np.arange(5)})

    def test_default_numpy(self):
        a = {
            'x': 5,
            'y': {
                'ya': 123.3,
                'yb': np.arange(5),
                'yc': [10, 20, 30],
                'yd': [10.0, 20.0, 30.0],
            },
        }
        a = TensorCollection(a, default_numpy=True)
        self.assertEqual(set(a.keys()), set(['x', 'y']))
        self.assertEqual(set(a['y'].keys()), set(['ya', 'yb', 'yc', 'yd']))
        self.assertEqual(a['x'].dtype, np.int64)
        self.assertEqual(a['y']['ya'].dtype, np.float64)
        self.assertEqual(a['y']['yb'].dtype, np.int64)
        self.assertEqual(a['y']['yc'].dtype, np.int64)
        self.assertEqual(a['y']['yd'].dtype, np.float64)
        self.assertArrayEqual(a['y']['yb'], [0, 1, 2, 3, 4])
        self.assertArrayEqual(a['y']['yc'], [10, 20, 30])
        self.assertArrayEqual(a['y']['yd'], [10.0, 20.0, 30.0])

    def test_unary_operators(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        a = TensorCollection(x=x)
        self.assertCollectionEqual(-a, {'x': -x})
        self.assertCollectionEqual(+a, {'x': +x})
        self.assertCollectionEqual(abs(a), {'x': abs(x)})

    def test_binary_operators(self):
        x = np.array([1, 2, 3, 4, 5])
        a = TensorCollection(x=x)
        self.assertCollectionEqual(a + 10, {'x': x + 10})
        self.assertCollectionEqual(a - 10, {'x': x - 10})
        self.assertCollectionEqual(a * 10, {'x': x * 10})
        self.assertCollectionEqual(a ** 3, {'x': x ** 3})
        self.assertCollectionEqual(a / 10, {'x': x / 10})
        self.assertCollectionEqual(a // 2, {'x': x // 2})
        self.assertCollectionEqual(a % 3, {'x': x % 3})
        self.assertCollectionEqual(a + a, {'x': x + x})
        self.assertCollectionEqual(a - a, {'x': x - x})
        self.assertCollectionEqual(a * a, {'x': x ** 2})
        self.assertCollectionEqual(a ** a, {'x': x ** x})
        self.assertCollectionEqual(a / a, {'x': x / x})
        self.assertCollectionEqual(a // a, {'x': x // x})
        self.assertCollectionEqual(a % a, {'x': x % x})

    def test_reverse_operators(self):
        x = np.array([1, 2, 3, 4, 5])
        a = TensorCollection(x=x)
        self.assertCollectionEqual(10 + a, {'x': 10 + x})
        self.assertCollectionEqual(10 - a, {'x': 10 - x})
        self.assertCollectionEqual(10 * a, {'x': 10 * x})
        self.assertCollectionEqual(10 ** a, {'x': 10 ** x})
        self.assertCollectionEqual(10 / a, {'x': 10 / x})
        self.assertCollectionEqual(10 // a, {'x': 10 // x})
        self.assertCollectionEqual(10 % a, {'x': 10 % x})

    def test_assignment_operators(self):
        x = np.array([1., 2., 3., 4., 5.])
        a = TensorCollection(x=x.copy())
        a += 10
        self.assertCollectionEqual(a, {'x': x + 10})
        a += x
        self.assertCollectionEqual(a, {'x': x + 10 + x})
        a *= x
        self.assertCollectionEqual(a, {'x': x * (x + 10 + x)})
        a /= x
        self.assertCollectionEqual(a, {'x': x + 10 + x})
        a %= 3
        self.assertCollectionEqual(a, {'x': (x + 10 + x) % 3})

    def test_binary_operator_collection_vs_array(self):
        x = np.array([1., 2., 3., 4., 5.])
        a = TensorCollection(xx=x)
        self.assertCollectionEqual(a + x, {'xx': x + x})
        self.assertCollectionEqual(x + a, {'xx': x + x})

    def test_binary_operator_broadcast(self):
        """Test broadcast for different hierarchies."""
        x = np.array([1., 2., 3., 4., 5.])
        a = TensorCollection(xx=dict(aa=x, bb=x), yy=x)
        b = TensorCollection(xx=x, yy=x)

        ab = {'xx': {'aa': x + x, 'bb': x + x}, 'yy': x + x}
        self.assertCollectionEqual(a + b, ab)
        self.assertCollectionEqual(b + a, ab)

    def test_torchlike_numpylike(self):
        x = np.array([[1., 2., 3.], [4., 5., 6.]])
        a = TensorCollection(x=x, y=x)
        self.assertCollectionEqual(a.sum(), {'x': 1+2+3+4+5+6, 'y': 1+2+3+4+5+6})
        self.assertCollectionEqual(a.sum(0), {'x': x.sum(0), 'y': x.sum(0)})
        self.assertCollectionEqual(a.mean(), {'x': 3.5, 'y': 3.5})
        self.assertCollectionEqual(a.mean(0), {'x': x.mean(0), 'y': x.mean(0)})

    def test_named_map(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        t = TensorCollection(aa=x, bb=dict(cc=y))

        # Test which keys are passed, test reconstruction.
        keys_list = []
        def fn(keys, v):
            keys_list.append(keys)
            return v[:2] if keys == ('aa',) else v[:3]
        self.assertCollectionEqual(
                t.named_map(fn), {'aa': [1, 2], 'bb': dict(cc=[1, 2, 3])})
        self.assertEqual(keys_list, [('aa',), ('bb', 'cc')])

        # Test None values are removed.
        self.assertCollectionEqual(t.named_map(lambda keys, v: None), {'bb': {}})

    def test_hierarchy(self):
        x = np.arange(5)
        t = TensorCollection(aa=x, bb=dict(cc=x))
        self.assertEqual(t.hierarchy(), {'aa': None, 'bb': {'cc': None}})
