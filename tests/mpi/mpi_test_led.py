from mpi_base import TestCase, MPI, world
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.led.topology import topology_split_comm
import adaled

import numpy as np


class DummyMicroSolver(adaled.MicroSolver):
    def __init__(self):
        self.dt = 0.1
        self.t = 0
        self.omega = 0.5

    def advance(self, F: np.ndarray):
        self.t += self.dt
        return self.get_state()

    def get_state(self):
        x = self.t * self.omega
        # 1 simulation, 2 state variables.
        return np.array([np.sin(x), np.cos(x)]).reshape(1, 2)


@dataclass
class DummyConfig(DataclassMixin):
    criteria = adaled.SimpleCriteriaConfig(
            k_warmup=5, k_cmp=24, max_macro_steps=100, max_micro_steps=50,
            max_cmp_error=1e-20, max_uncertainty=1e-20)
    rnn = adaled.RNNConfig(input_size=2, output_size=2, rnn_hidden_size=4)
    ensemble_size = 5
    server = adaled.AdaLEDServerConfig(
            quiet=True, init_output_folder=False,
            dump_diagnostics=adaled.DumpConfig(every=0))
    client = adaled.AdaLEDClientConfig(max_steps=100, quiet=True)


class MPITestServerClient(TestCase):
    def _test_server_client(self, config: DummyConfig, num_ranks: int, topology: str):
        adaled.init_torch()

        comm = self.prepare_or_skip(num_ranks)
        topology = topology_split_comm(comm, topology)
        assert topology.is_adaled_active  # function does not support non-active ranks

        rnns = [adaled.RNN(config.rnn) for _ in range(config.ensemble_size)]
        macro = adaled.DeterministicPropagatorEnsemble(rnns)

        if topology.is_server:
            datasets = adaled.utils.data.collections \
                    .CappedFixedLengthTrajectoryDatasetCollection(
                        1000, 10, trajectory_length=30)
            server = adaled.AdaLEDServer(
                    config.server, macro, datasets, topology=topology)
            server.run()
        else:
            client = adaled.AdaLEDClient(
                    config.client, macro, config.criteria.create(), topology=topology)
            generator = client.make_generator(DummyMicroSolver())
            client.run(generator)

    def test_server_client_2_3(self):
        self._test_server_client(DummyConfig(), 5, 'S0,S1,C0->S0,C1->S1,C2->S0+S1')

    def test_server_client_3_1(self):
        "Test more servers than clients."""
        self._test_server_client(DummyConfig(), 4, 'S0,S1,S2,C0->S0+S1+S2')
