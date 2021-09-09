from base import TestCase
from adaled import TensorCollection
import adaled.utils.data.collections as collections
import adaled.utils.data.datasets as datasets

import numpy as np

import os
import tempfile
import unittest.mock

class UniformTrajectoryLengthDatasetTests:
    SUPPORTS_PURE_NDARRAY = True

    def setUp(self):
        super().setUp()
        self.dataset = None

    def tearDown(self):
        super().tearDown()
        self.dataset = None

    def test_add_trajectories_ndarray(self):
        if not self.SUPPORTS_PURE_NDARRAY:
            self.skipTest("does not support pure ndarray trajectories")
        d = self.dataset
        # Add four trajectories of length 5, each state being a 1x1 matrix.
        d.add_trajectories(1.0 * np.arange(10).reshape(2, 5, 1, 1))
        d.add_trajectories(2.0 * np.arange(10).reshape(2, 5, 1, 1))

        expected = np.array([
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
            [0., 2., 4., 6., 8.],
            [10., 12., 14., 16., 18.],
        ])[:, :, np.newaxis, np.newaxis]
        self.assertArrayEqual(d.as_trajectories(), expected)
        self.assertArrayEqual(d.as_states(), expected.reshape(20, 1, 1))

    def test_add_trajectories_tensor_collection(self):
        d = self.dataset
        # Add two trajectories of length 5, each state being a 1x1 matrix.
        d.add_trajectories(TensorCollection(
                {'xy': {'ab': np.arange(10).reshape(2, 5, 1, 1)}}))

        second = np.arange(5, 10)[:, np.newaxis, np.newaxis]
        self.assertCollectionEqual(d.as_trajectories()[1], {'xy': {'ab': second}})
        self.assertCollectionEqual(d.as_trajectories('xy')[1], {'ab': second})
        self.assertArrayEqual(d.as_trajectories(('xy', 'ab'))[1],  second)

        self.assertCollectionEqual(d.as_states()[5+2], {'xy': {'ab': second[2]}})
        self.assertCollectionEqual(d.as_states('xy')[5+2], {'ab': second[2]})
        self.assertArrayEqual(d.as_states(('xy', 'ab'))[5+2], second[2])

    def test_update_states(self):
        d = self.dataset
        # Add two trajectories of length 5, each state being a 1x1 matrix.
        d.add_trajectories(TensorCollection(
                {'xy': {'ab': np.arange(10).reshape(2, 5, 1, 1)}}))

        # Try modifying one array.
        d.update_states([1, 7, 8], ('xy', 'ab'), [[[-1]], [[-2]], [[-3]]])
        computed = np.array(d.as_states(('xy', 'ab'))).reshape(10)
        expected = np.array([0, -1, 2, 3, 4, 5, 6, -2, -3, 9])
        self.assertArrayEqual(computed, expected)

        # Try modifying a tensor collection.
        d.update_states(
                [1, 7, 8], ('xy',),
                TensorCollection(ab=np.asarray([-10, -20, -30]).reshape(3, 1, 1)))
        computed = np.array(d.as_states(('xy', 'ab'))).reshape(10)
        expected = np.array([0, -10, 2, 3, 4, 5, 6, -20, -30, 9])
        self.assertArrayEqual(computed, expected)

    def test_update_trajectories(self):
        d = self.dataset
        # Add two trajectories of length 5, each state being a 1x1 matrix.
        d.add_trajectories(TensorCollection(
                {'xy': {'ab': np.arange(10).reshape(2, 5, 1, 1)}}))

        # Try modifying one array.
        ab = np.broadcast_to(
                np.array([-1, -2])[:, np.newaxis, np.newaxis, np.newaxis],
                (2, 5, 1, 1))
        d.update_trajectories([1, 0], ('xy', 'ab'), ab)
        computed = np.array(d.as_trajectories(('xy', 'ab'))).reshape(2, 5)
        expected = np.array([[-2, -2, -2, -2, -2], [-1, -1, -1, -1, -1]])
        self.assertArrayEqual(computed, expected)

        # Try modifying a tensor collection.
        d.update_trajectories([1, 0], ('xy',), TensorCollection(ab=10 * ab))
        computed = np.array(d.as_trajectories(('xy', 'ab'))).reshape(2, 5)
        expected = np.array([[-20, -20, -20, -20, -20], [-10, -10, -10, -10, -10]])
        self.assertArrayEqual(computed, expected)

class TestUniformInMemory(TestCase, UniformTrajectoryLengthDatasetTests):
    def setUp(self):
        super().setUp()
        self.dataset = datasets.UniformInMemoryTrajectoryDataset()


class TestCappedDataset(TestCase):
    def test_capped_dataset(self):
        # Add 3x two trajectories into a dataset of capacity 5. The final
        # trajectory replaces the trajectory #2.
        class MockPolicy(datasets.ReplacementPolicy):
            def __init__(self):
                self.queries = []

            def add_trajectories(self, batch):
                self.queries.append(batch)

            def replace(self, batch):
                self.queries.append(batch)
                return [2] * len(batch)  # Trajectory to replace.

        policy = MockPolicy()
        dataset = datasets.UniformInMemoryTrajectoryDataset()
        dataset = datasets.CappedTrajectoryDataset(dataset, policy, capacity=5)

        dataset.add_trajectories(1.0 * np.arange(10).reshape(2, 5))
        dataset.add_trajectories(2.0 * np.arange(10).reshape(2, 5))
        dataset.add_trajectories(3.0 * np.arange(10).reshape(2, 5))

        expected = 1.0 * np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [15, 18, 21, 24, 27],  # <-- Replaced at the end.
            [10, 12, 14, 16, 18],
            [0, 3, 6, 9, 12],
        ])
        self.assertArrayEqual(dataset.as_trajectories(), expected)

        expected_queries = [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            np.empty((0, 5)),
            [[0, 2, 4, 6, 8], [10, 12, 14, 16, 18]],
            np.empty((0, 5)),
            [[0, 3, 6, 9, 12]],  # <-- Last batch was split.
            [[15, 18, 21, 24, 27]],
        ]
        for query, expected in zip(policy.queries, expected_queries):
            self.assertArrayEqual(query, expected)


class TestFixedLengthTrajectoryDataset(TestCase):
    def test_add_trajectories_nonuniform_length(self):
        parent = datasets.UniformInMemoryTrajectoryDataset()
        d = datasets.FixedLengthTrajectoryDataset(parent, trajectory_length=16)

        # Test implementation details.
        def mock_randint(begin, end, n):
            if (begin, end, n) == (0, 12, 2):
                return [5, 8]
            elif (begin, end, n) == (0, 2, 1):
                return [1]
            else:
                raise Exception(f"unexpected input: {begin} {end} {n}")

        with unittest.mock.patch('numpy.random.randint', mock_randint):
            d.add_trajectories([
                TensorCollection(xy=np.arange(27)),
                TensorCollection(xy=np.arange(17)),
            ])

        traj = d.as_trajectories()
        self.assertEqual(len(traj), 3)
        self.assertCollectionEqual(traj[0], {'xy': np.arange(5, 5 + 16)})
        self.assertCollectionEqual(traj[1], {'xy': np.arange(8, 8 + 16)})
        self.assertCollectionEqual(traj[2], {'xy': np.arange(1, 1 + 16)})

class TestDatasetIO(TestCase):
    def _test_io(self, dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'dataset.pt')
            dataset.save(path, verbose=False)

            loaded = datasets.load_dataset(path)
            self.assertIs(dataset.__class__, loaded.__class__)
            self.assertArrayEqual(dataset.as_states(), loaded.as_states())
            self.assertArrayEqual(dataset.as_trajectories(), loaded.as_trajectories())

    def test_uniform_in_memory_io(self):
        dataset = datasets.UniformInMemoryTrajectoryDataset()
        # Add two trajectories of length 5, each state being a vector of size 4.
        dataset.add_trajectories(1.0 * np.arange(40).reshape(2, 5, 4))
        self._test_io(dataset)

    def test_uniform_fixed_length_capped_collection(self):
        # Testing the convenience class.
        collection = collections.CappedFixedLengthTrajectoryDatasetCollection(
                train_capacity=10, valid_capacity=6, trajectory_length=10)
        # 15x add 2 trajectories of length 20 with 5x5 matrices as states.
        for i in range(15):
            collection.add_trajectories(np.random.uniform(-1.0, 1.0, (2, 20, 5, 5)))

        with tempfile.TemporaryDirectory() as tmpdir:
            collection.save(tmpdir, verbose=False)
            loaded = collections.DynamicTrajectoryDatasetCollection.load(tmpdir)
            self.assertEqual(collection.train_portion, loaded.train_portion)
            self.assertArrayEqual(collection.train_dataset.as_trajectories(),
                                  loaded.train_dataset.as_trajectories())
            self.assertArrayEqual(collection.valid_dataset.as_states(),
                                  loaded.valid_dataset.as_states())
