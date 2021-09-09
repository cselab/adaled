from base import TestCase

from adaled.backends.generic import extended_emptylike
from adaled.backends.soa import TensorCollection, TCShape
import numpy as np

class TestGeneric(TestCase):
    def test_extended_emptylike_numpy(self):
        a = np.zeros((3, 4, 5))
        fn = extended_emptylike
        self.assertShape(fn(a, 10), (10, 3, 4, 5))
        self.assertShape(fn(a, 10, axis=1), (3, 10, 4, 5))
        self.assertShape(fn(a, 10, axis=-1), (3, 4, 5, 10))
        self.assertShape(fn(a, 10, axis=-2), (3, 4, 10, 5))
        self.assertShape(fn(a, (10, 15)), (10, 15, 3, 4, 5))
        self.assertShape(fn(a, (10, 15), axis=1), (3, 10, 15, 4, 5))
        self.assertShape(fn(a, (10, 15), axis=-1), (3, 4, 5, 10, 15))
        self.assertShape(fn(a, (10, 15), axis=-2), (3, 4, 10, 15, 5))

    def test_extended_emptylike_tensor_collection(self):
        a = TensorCollection(xx=np.zeros((3, 4, 5)), yy=np.zeros((3, 4)))
        fn = extended_emptylike
        self.assertShape(fn(a, 10), {'xx': (10, 3, 4, 5), 'yy': (10, 3, 4)})
        self.assertShape(fn(a, 10, axis=1), {'xx': (3, 10, 4, 5), 'yy': (3, 10, 4)})
        self.assertShape(fn(a, 10, axis=-1), {'xx': (3, 4, 5, 10), 'yy': (3, 4, 10)})
        self.assertShape(fn(a, 10, axis=-2), {'xx': (3, 4, 10, 5), 'yy': (3, 10, 4)})
        self.assertShape(fn(a, (10, 15)), {'xx': (10, 15, 3, 4, 5), 'yy': (10, 15, 3, 4)})
        self.assertShape(fn(a, (10, 15), axis=1), {'xx': (3, 10, 15, 4, 5), 'yy': (3, 10, 15, 4)})
        self.assertShape(fn(a, (10, 15), axis=-1), {'xx': (3, 4, 5, 10, 15), 'yy': (3, 4, 10, 15)})
        self.assertShape(fn(a, (10, 15), axis=-2), {'xx': (3, 4, 10, 15, 5), 'yy': (3, 10, 15, 4)})
