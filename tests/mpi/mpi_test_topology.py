from mpi_base import TestCase, world
from adaled.led.topology import topology_split_comm

import time

class MPITestTopology(TestCase):
    def test_1S_1C(self):
        comm = self.prepare_or_skip(2)
        t = topology_split_comm(comm, 'S0,C0->S0')
        self.assertComm(t.world, 2, comm.rank, False)
        self.assertComm(t.active_comm, 2, comm.rank, False)
        self.assertComm(t.active_side_comm, 1, 0, False)
        self.assertComm(t.active_intercomm, 1, 0, True, remote_size=1)
        self.assertComm(t.component_comm, 1, 0, False)
        self.assertEqual(t.remote_activecomm_ranks, [0])

    def test_1S_1C_x3(self):
        comm = self.prepare_or_skip(6)
        t = topology_split_comm(comm, 'S0,C0->S0')
        self.assertComm(t.world, 6, comm.rank, False)
        self.assertComm(t.active_comm, 6, comm.rank, False)
        self.assertComm(t.active_side_comm, 3, comm.rank // 2, False)
        self.assertComm(t.active_intercomm, 3, comm.rank // 2, True, remote_size=3)
        self.assertComm(t.component_comm, 1, 0, False)
        self.assertEqual(t.remote_activecomm_ranks, [comm.rank // 2])

    def test_3S_3C_all_to_all(self):
        comm = self.prepare_or_skip(6)
        t = topology_split_comm(
                comm, 'S0,C0->S0+S1+S2,S1,C1->S0+S1+S2,S2,C2->S0+S1+S2')
        self.assertComm(t.world, 6, comm.rank, False)
        self.assertComm(t.active_comm, 6, comm.rank, False)
        self.assertComm(t.active_side_comm, 3, comm.rank // 2, False)
        self.assertComm(t.active_intercomm, 3, comm.rank // 2, True, remote_size=3)
        self.assertComm(t.component_comm, 1, 0, False)
        self.assertEqual(t.remote_activecomm_ranks, [0, 1, 2])

    def test_3S_3C_cyclic(self):
        comm = self.prepare_or_skip(6)
        t = topology_split_comm(comm, 'S0,C0->S0+S1,S1,C1->S1+S2,S2,C2->S2+S0')
        self.assertComm(t.world, 6, comm.rank, False)
        self.assertComm(t.active_comm, 6, comm.rank, False)
        self.assertComm(t.active_side_comm, 3, comm.rank // 2, False)
        self.assertComm(t.active_intercomm, 3, comm.rank // 2, True, remote_size=3)
        self.assertComm(t.component_comm, 1, 0, False)
        if t.is_server:
            self.assertEqual(
                    sorted(t.remote_activecomm_ranks),
                    sorted([comm.rank // 2, (comm.rank // 2 - 1 + 3) % 3]))
        else:
            self.assertEqual(t.remote_activecomm_ranks,
                             [comm.rank // 2, (comm.rank // 2 + 1) % 3])

    def test_1S_1CC_x2(self):
        # Ranks are grouped in groups of size 3, of which 0th rank is the
        # server, 1st is the active client, and 2nd is inactive rank (not
        # participating in recording the trajectory or evaluating the
        # networks).
        comm = self.prepare_or_skip(6)
        t = topology_split_comm(comm, 'S0,C0->S0,C0')
        self.assertComm(t.world, 6, comm.rank, False)
        mod = comm.rank % 3
        if mod <= 1:
            rank_active = comm.rank // 3 * 2 + mod
            self.assertComm(t.active_comm, 4, rank_active, False)
            self.assertComm(t.active_side_comm, 2, comm.rank // 3, False)
            self.assertComm(t.active_intercomm, 2, comm.rank // 3, True, remote_size=2)
            self.assertComm(t.component_comm, (1 if mod == 0 else 2), 0, False)
        else:
            self.assertCommNull(t.active_comm)
            self.assertCommNull(t.active_side_comm)
            self.assertCommNull(t.active_intercomm)
            self.assertComm(t.component_comm, 2, 1, False)

    def test_at_least_one_server(self):
        with self.assertRaises(ValueError):
            topology_split_comm(world, 'C0,C1')

    def test_at_least_one_client(self):
        with self.assertRaises(ValueError):
            topology_split_comm(world, 'S0,S1')

    def test_servers_do_not_repeat(self):
        with self.assertRaises(ValueError):
            topology_split_comm(world, 'S0,C0,S0')

    def test_first_client_rank_specifies_servers(self):
        with self.assertRaises(ValueError):
            topology_split_comm(world, 'S0,C0')

        with self.assertRaises(ValueError):
            topology_split_comm(world, 'S0,C0->S0,C0->S0')
