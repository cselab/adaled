from mpi_base import TestCase, MPI, world
from adaled import TensorCollection
from adaled.conduits.mpi import IntercommConduit

import numpy as np

from typing import Callable
import os
import time

def make_string_factory(size_bytes: int):
    def make_string(source: int, target: int):
        """Create string data of approx size_bytes bytes."""
        s = f'[{source} to {target}]'
        s = s * ((size_bytes + len(s) - 1) // len(s))
        return s

    return make_string

def make_tensor_collection_factory(size_bytes: int):
    n = size_bytes // (4 + 4 + 8)

    def make_tc(source: int, target: int) -> TensorCollection:
        """Create a TensorCollection of approx size_bytes bytes."""
        a = np.full(n, source, dtype=np.int32)
        b = np.full(n, source * target, dtype=np.float32)
        c = np.full(n, target, dtype=np.float64)
        return TensorCollection({'ab': {'a': a, 'b': b}, 'c': c})

    return make_tc

class MPITestConduit(TestCase):
    def _test_message_exchange(
            self,
            num_servers: int,
            num_clients: int,
            package_factory: Callable,
            assert_func: Callable):
        comm, localcomm, intercomm, is_server = \
                self.prepare_intercomm_or_skip(num_servers, num_clients)

        conduit = IntercommConduit(intercomm)
        conduit.start()

        if is_server:
            for remote_rank in range(num_clients):
                conduit.send(package_factory(localcomm.rank, remote_rank),
                             remote_rank)

        conduit.wait_and_close()

        if not is_server:
            messages = conduit.pop_all()
            self.assertEqual(len(messages), num_servers)
            self.assertEqual(set(message.rank for message in messages),
                             set(range(num_servers)))
            for message in messages:
                expected = package_factory(message.rank, localcomm.rank)
                assert_func(message.data, expected)

    def test_message_exchange_small_data(self):
        self._test_message_exchange(2, 2, make_string_factory(10), self.assertEqual)

    def test_message_exchange_large_data(self):
        self._test_message_exchange(2, 2, make_string_factory(1000000), self.assertEqual)

    def test_message_exchange_huge_tensor_collection(self):
        """Test the transmission of very large data. Use the environment
        variable ADALED_TEST_MESSAGE_BYTES to control the size.

        The current implementation serializes the whole packages and thus fails
        for 2-4GB or larger packages. The serialization should be avoided if
        possible (although we have to be careful with copying vs non-copying
        and API guarantees), and some kind of chunking should be implemented to
        allow for larger packages.
        """
        size_bytes = os.environ.get('ADALED_TEST_MESSAGE_BYTES', None)
        if size_bytes is not None:
            size_bytes = int(size_bytes)
            print(f"Using message size of {size_bytes / 1024 / 1024:.2f} MB.", flush=True)
        else:
            size_bytes = 2000000
        self._test_message_exchange(1, 1, make_tensor_collection_factory(size_bytes),
                                    self.assertCollectionEqual)
