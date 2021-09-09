from mpi_base import TestCase
import adaled.utils.data.datasets as datasets

import numpy as np


class MPITestCappedDataset(TestCase):
    def test_add_trajectories_is_collective(self):
        comm = self.prepare_or_skip(3)

        class MockDataset(datasets.UniformInMemoryTrajectoryDataset):
            def __init__(self):
                super().__init__()
                self.queries = []

            def add_trajectories(self, batch):
                # comm.Barrier()  # If it deadlocks, something is wrong.
                self.queries.append(('ADD', batch))
                super().add_trajectories(batch)

            def set_trajectories(self, indices, batch):
                self.queries.append(('SET', indices, batch))
                super().set_trajectories(indices, batch)

        class MockPolicy(datasets.ReplacementPolicy):
            def add_trajectories(self, batch):
                pass

            def replace(self, batch):
                return list(range(2, 2 + len(batch)))  # Trajectories to replace.

        storage = MockDataset()
        policy = MockPolicy()
        dataset = datasets.CappedTrajectoryDataset(storage, policy, 10, comm)

        def make(duration):
            # All trajectories have 6 states, each 7-dim state.
            return np.zeros((duration, 6, 7))

        if comm.rank == 0:
            dataset.add_trajectories(make(7))
            dataset.add_trajectories(make(3))  # Reach capacity.
            dataset.add_trajectories(make(2))  # Still calls add_traj.
            dataset.add_trajectories(make(2))
        elif comm.rank == 1:
            dataset.add_trajectories(make(0))
            dataset.add_trajectories(make(8))
            dataset.add_trajectories(make(5))  # Now everyone is full.
            dataset.add_trajectories(make(2))
        else:
            dataset.add_trajectories(make(7))
            dataset.add_trajectories(make(5))
            dataset.add_trajectories(make(3))
            dataset.add_trajectories(make(3))

        # Test storage.
        qs = storage.queries
        if comm.rank == 0:
            expected = [
                ('ADD', make(7)),
                ('SET', [], make(0)),
                ('ADD', make(3)),
                ('SET', [], make(0)),
                ('ADD', make(0)),
                ('SET', [2, 3], make(2)),
                ('SET', [2, 3], make(2)),
            ]
        elif comm.rank == 1:
            expected = [
                ('ADD', make(0)),
                ('SET', [], make(0)),
                ('ADD', make(8)),
                ('SET', [], make(0)),
                ('ADD', make(2)),
                ('SET', [2, 3, 4], make(3)),
                ('SET', [2, 3], make(2)),
            ]
        else:
            expected = [
                ('ADD', make(7)),
                ('SET', [], make(0)),
                ('ADD', make(3)),
                ('SET', [2, 3], make(2)),
                ('ADD', make(0)),
                ('SET', [2, 3, 4], make(3)),
                ('SET', [2, 3, 4], make(3)),
            ]

        self.assertEqual(len(qs), len(expected))
        for q, ex in zip(qs, expected):
            self.assertEqual(q[0], ex[0])
            self.assertArrayEqual(q[1], ex[1])
            if q[0] == 'SET':
                self.assertArrayEqual(q[2], ex[2])
