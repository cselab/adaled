from base import TestCase
from adaled.backends import TensorCollection
from adaled.utils.buffer import DynamicArray

import numpy as np

import pickle

class TestDynamicArray(TestCase):
    def test_empty(self):
        x = DynamicArray()
        self.assertEqual(len(x), 0)
        self.assertEqual(list(x), [])
        self.assertEqual(len(x.data), 0)
        self.assertEqual(list(x.data), [])

        # Slice of an empty sequence is an empty sequence.
        self.assertEqual(list(x[:]), [])
        self.assertEqual(list(x[1:10:3]), [])

        # Taking empty tuple of an empty sequence is an empty sequence.
        self.assertEqual(list(x[()]), [])

        with self.assertRaises(IndexError):
            x[5]

        with self.assertRaises(TypeError):
            x[5, 3]  # Unsupported, just throw something.

    def test_like(self):
        template = np.zeros((3, 4))
        x = DynamicArray(like=template)
        assert len(x) == 0
        assert x.shape == (0, 3, 4)

    def test_append_python_float(self):
        """Appending a Python float raises an unrecognized backend error."""
        x = DynamicArray()
        with self.assertRaisesRegex(TypeError, r".*object is not subscriptable.*"):
            x.append(10.0)  # Cannot determine storage backend.

        x.append(np.float32(20.0))
        x.append(30.0)  # Now an unspecified type is fine.
        self.assertArrayEqualStrict(x.data, np.array([20., 30.], dtype=np.float32))

    def test_append_python_int(self):
        """Appending a Python int raises an unrecognized backend error."""
        x = DynamicArray()
        with self.assertRaisesRegex(TypeError, r".*object is not subscriptable.*"):
            x.append(10)  # Cannot determine storage backend.

        x.append(np.int32(20))
        x.append(30)  # Now an unspecified type is fine.
        self.assertArrayEqualStrict(x.data, np.array([20, 30], dtype=np.int32))

    def test_append_int32(self):
        x = DynamicArray()
        x.append(np.int32(123))
        self.assertArrayEqual(x.data, [123])
        self.assertEqual(x.data.dtype, np.int32)

    def test_append_scalar(self):
        x = DynamicArray()
        x.append(np.array(10.0))
        x.append(np.array(20.0))
        self.assertArrayEqual(x.data, [10.0, 20.0])

    def test_append_tensor(self):
        x = DynamicArray()
        x.append(np.arange(12).reshape(3, 4))
        x.append(12 + np.arange(12).reshape(3, 4))
        self.assertEqual(x.data.shape, (2, 3, 4))
        self.assertArrayEqual(x.data.ravel(), np.arange(24))

    def test_append_tensor_then_scalar(self):
        """x.append(np.nan) should work even when data is not a scalar."""
        x = DynamicArray()
        x.append(TensorCollection(a=np.arange(4.), b=np.arange(3.)))
        x.append(-1.0)
        self.assertCollectionEqual(
                x.data, {
                    'a': [[0.0, 1.0, 2.0, 3.0], [-1.0, -1.0, -1.0, -1.0]],
                    'b': [[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]],
                })

    def test_tensor_collection(self):
        item = TensorCollection(a=np.arange(5), b=np.arange(8).reshape((2, 4)))
        x = DynamicArray()
        x.append(item)
        x.append(item)
        x.append(item)
        self.assertEqual(x['a'].shape, (3, 5))
        self.assertEqual(x['b'].shape, (3, 2, 4))

    def test_array_interface(self):
        x = DynamicArray()
        n = 0
        while n < 10 or len(x._buffer) == len(x):
            x.append(np.int32(n))
            n += 1
            if n == 12:
                self.fail("buffer does not allocate in advance")
        self.assertArrayEqual(x, np.arange(n, dtype=np.int32))
        self.assertArrayEqualStrict(np.asarray(x), np.arange(n, dtype=np.int32))

        np.asarray(x)[0] = -1
        self.assertEqual(x[0], -1)

    def test_pickle_empty(self):
        # Test uninitialized empty.
        x = DynamicArray()
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(len(y), 0)
        self.assertEqual(len(y.data), 0)
        y.extend(np.zeros((10, 3), dtype=np.float32))
        self.assertEqual(y.data.shape, (10, 3))
        self.assertEqual(y.data.dtype, np.float32)

        # Test initialized empty.
        x = DynamicArray()
        x.append(np.zeros((3, 4), dtype=np.float32))
        x.clear()
        y = pickle.loads(pickle.dumps(x))
        y.reserve(10)  # The type information is already known.
        self.assertEqual(y._buffer.shape, (10, 3, 4))
        self.assertEqual(y._buffer.dtype, np.float32)

    def test_pickle_nonempty(self):
        x = DynamicArray()
        x.append(np.arange(10))
        x.append(np.arange(10))
        x.append(np.arange(10))

        # Test restoring works.
        s = pickle.dumps(x)
        y = pickle.loads(s)
        self.assertEqual(len(y.data), 3)
        self.assertArrayEqual(x.data, y.data)

        x.reserve(100000)
        s = pickle.dumps(x)
        self.assertLess(len(s), 10000)  # Dump must not contain the reserved part.
