import time

class MockMPIComm:
    """Mock communicator used when no MPI communicator is provided."""
    @property
    def size(self):
        return 1

    @property
    def rank(self):
        return 0

    def Allreduce(self, local_, global_):
        assert local_.shape == global_.shape
        assert local_.dtype == global_.dtype
        global_[:] = local_


mock_world = MockMPIComm()


def cpu_friendly_barrier(comm: 'mpi4py.MPI.Comm'):
    """Barrier that sleeps while waiting instead of running a busy loop."""
    r = comm.Ibarrier()
    while True:
        if r.Test():
            return
        time.sleep(0.001)
