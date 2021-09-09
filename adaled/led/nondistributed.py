from adaled.conduits.base import make_single_rank_conduit_pair
from adaled.led.client import AdaLEDClient, AdaLEDClientConfig, AdaLEDGenerator
from adaled.led.criteria import AdaLEDCriteria
from adaled.led.recording import RecorderConfig, TrajectoryRecorder
from adaled.led.server import AdaLEDServer, AdaLEDServerConfig
from adaled.led.trainers import LEDTrainer
from adaled.solvers.base import MicroSolver, MacroSolver
from adaled.transformers.base import Transformer, IdentityTransformer
from adaled.utils.data.collections import DynamicTrajectoryDatasetCollection
from adaled.utils.dataclasses_ import dataclass

from typing import Any, Dict, Optional, Union
import dataclasses
import sys
import time
import warnings

@dataclass
class AdaLEDConfig(AdaLEDServerConfig, AdaLEDClientConfig):
    pass


class AdaLED:
    """Non-distributed AdaLED or distributed AdaLED where server and client
    side alternate each cycle.

    For distributed runs, prefer using AdaLEDClient and AdaLEDServer manually.
    """
    def __init__(
            self,
            macro: MacroSolver,
            config: AdaLEDConfig,
            criteria: AdaLEDCriteria,
            datasets: DynamicTrajectoryDatasetCollection,
            trainer: Optional[LEDTrainer] = None,
            transformer: Transformer = IdentityTransformer(),
            server_cls: type = AdaLEDServer,
            client_cls: type = AdaLEDClient,
            server_kwargs: Dict[str, Any] = {},
            client_kwargs: Dict[str, Any] = {},
            comm=None):
        if config.always_run_micro and config.compensate_for_stats_overhead:
            warnings.warn("always_run_micro is set, ignoring "
                          "compensate_for_stats_overhead in non-distributed run")
        self.server = server_cls(
                config=config, macro=macro, transformer=transformer,
                datasets=datasets, trainer=trainer, comm=comm,
                **server_kwargs)
        self.client = client_cls(
                config=config, macro=macro, transformer=transformer,
                criteria=criteria, comm=comm, **client_kwargs)

    @property
    def diagnostics(self):
        return self.server.diagnostics

    def make_generator(self, *args, **kwargs):
        return self.client.make_generator(*args, **kwargs)

    def run(self,
            generator: AdaLEDGenerator,
            recorder: Union[TrajectoryRecorder, RecorderConfig] = RecorderConfig()):
        """Execute AdaLED by alternating between client and server.

        Perform one cycle of AdaLED in the client and then one round of training in
        the server. This represents the basic non-distributed or synchronized
        distributed execution schedule between client and server.
        """
        client = self.client
        server = self.server

        server_conduit, client_conduit = make_single_rank_conduit_pair()
        server_runner = server.make_runner(conduit=server_conduit)
        client_runner = client.make_runner(generator, recorder, conduit=client_conduit)
        server_runner.start()
        client_runner.start()

        t0 = time.time()
        while True:
            # TODO: Test if restart works and causes no deadlock when client
            # decides to stop the simulation just before the server decides to
            # save restart files.
            server_done = server_runner.step(no_sleep=True)
            client_done = client_runner.run_cycle()

            if server_done or client_done:
                server_runner.close()
                client_runner.close()
                break
            sys.stdout.flush()
            sys.stderr.flush()
        print("Total execution time:", time.time() - t0)
