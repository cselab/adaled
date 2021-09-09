from base import TestCase
from adaled import backends
from adaled.backends import TensorCollection, get_backend
from adaled.led import AdaLEDStage as Stage
from adaled.postprocessing.record import load_and_concat_records
from adaled.postprocessing.run import PostprocessRunner
import adaled

import numpy as np

import os
import shutil

DIR = os.path.dirname(os.path.abspath(__file__))

class _Micro(adaled.MicroSolver):
    def __init__(self, backend, complex_state: bool):
        dtype = backend.module.float32
        if complex_state:
            x = backend.zeros((1, 4), dtype=dtype)
            self.x = adaled.TensorCollection({
                'aa': x,
                'bb': {
                    'cc': 0 + x,  # Copy.
                },
            })
        else:
            self.x = backend.zeros((1, 8), dtype=dtype)

        self._step = 0

    def state_dict(self) -> dict:
        return {
            'step': self._step,
            'state': self.x,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step = state['step']
        self.x = state['state']

    def advance(self, F: np.ndarray):
        self._step += 1
        self.x += self._step
        return self.get_state()

    def get_state(self):
        return self.x

    def update_state(self, new_state, skip_steps):
        self._step += skip_steps
        self.x = 0 + new_state  # Copy.
        assert self.get_state().shape == new_state.shape


class _Transformer(adaled.Transformer):
    def __init__(self, backend, complex_state: bool):
        self.backend = backend
        self.complex_state = complex_state

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}

    def transform(self, x: TensorCollection):
        if self.complex_state:
            return self.backend.cat(list(x.allvalues()), axis=-1)
        else:
            return x

    def inverse_transform(self, z):
        if self.complex_state:
            return adaled.TensorCollection({
                'aa': z[..., :4],
                'bb': {
                    'cc': z[..., 4:],
                },
            })
        else:
            return z


class _TestE2ECase(TestCase):
    def setUp(self):
        self.old_dir = os.path.abspath(os.getcwd())
        dir = os.path.join(DIR, 'output_e2e')
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)

    def tearDown(self):
        os.chdir(self.old_dir)


class TestE2ESimpleSolver(_TestE2ECase):
    def _test(self, backend, complex_state: bool):
        micro = _Micro(backend, complex_state)
        transformer = _Transformer(backend, complex_state)

        rnn = adaled.RNNConfig(input_size=8, output_size=8, append_F=False)
        macro = adaled.DeterministicPropagatorEnsemble([adaled.RNN(rnn) for _ in range(5)])

        config = adaled.AdaLEDConfig(
                max_steps=100,
                quiet=True,
                dump_dataset=adaled.DumpConfig(every=0),
                dump_macro=adaled.DumpConfig(every=0),
                dump_transformer=adaled.DumpConfig(every=0),
                dump_diagnostics=adaled.DumpConfig(every=0))

        datasets = adaled.utils.data.collections.CappedFixedLengthTrajectoryDatasetCollection(
            train_capacity=100, valid_capacity=10, trajectory_length=14)

        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=3, k_cmp=10, max_micro_steps=(15, 20),
                max_macro_steps=(30, 40), num_relaxation_steps=3)
        recorder = adaled.RecorderConfig(start_every=10000, num_steps=10000)

        led = adaled.AdaLED(macro, config, criteria.create(), datasets,
                            transformer=transformer)
        led.run(led.make_generator(micro), recorder)

    def test_simple_state_numpy(self):
        self._test(backend=backends.backend_numpy.NumpyBackend, complex_state=False)

    def test_complex_state_numpy(self):
        self._test(backend=backends.backend_numpy.NumpyBackend, complex_state=True)

    def test_simple_state_torch_cpu(self):
        self._test(backend=backends.backend_torch.TorchBackend('cpu'), complex_state=False)

    def test_complex_state_torch_cpu(self):
        self._test(backend=backends.backend_torch.TorchBackend('cpu'), complex_state=True)


class TestE2ERestartMechanism(_TestE2ECase):
    def test_restart(self, backend=backends.backend_numpy.NumpyBackend, complex_state=False):
        micro1 = _Micro(backend, complex_state)
        micro2 = _Micro(backend, complex_state)
        transformer = _Transformer(backend, complex_state)

        rnn = adaled.RNNConfig(input_size=8, output_size=8, append_F=False)
        macro = adaled.DeterministicPropagatorEnsemble([adaled.RNN(rnn) for _ in range(5)])

        config = adaled.AdaLEDConfig(
                max_steps=10000,
                quiet=False,
                dump_dataset=adaled.DumpConfig(every=0),
                dump_macro=adaled.DumpConfig(every=0),
                dump_transformer=adaled.DumpConfig(every=0),
                dump_diagnostics=adaled.DumpConfig(every=0),
                max_seconds=1e-10,
                restart=False)

        datasets = adaled.utils.data.collections.CappedFixedLengthTrajectoryDatasetCollection(
            train_capacity=100, valid_capacity=10, trajectory_length=1 + 3 + 10 + 15)

        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=3, k_cmp=10, max_micro_steps=15,
                max_macro_steps=40, num_relaxation_steps=3,
                max_cmp_error=-1e10)  # Never accept.
        recorder_config = adaled.RecorderConfig(start_every=10000, num_steps=10000)

        # This should complete only one cycle due to the time limit.
        led1 = adaled.AdaLED(macro, config, criteria.create(), datasets,
                             transformer=transformer)
        recorder1 = adaled.TrajectoryRecorder(recorder_config)
        led1.run(led1.make_generator(micro1), recorder1)

        # So far there should be 1 trajectory in the dataset.
        num1 = len(led1.server.datasets.train_dataset.as_trajectories())
        num2 = len(led1.server.datasets.valid_dataset.as_trajectories())
        self.assertEqual(num1 + num2, 1)

        config.restart = True
        led2 = adaled.AdaLED(macro, config, criteria.create(), datasets,
                             transformer=transformer)
        recorder2 = adaled.TrajectoryRecorder(recorder_config)
        led2.run(led2.make_generator(micro2), recorder2)

        # Test whether the old recorder state was recovered.
        self.assertEqual(len(recorder1.record_x), 1 * (3 + 10 + 15))
        self.assertEqual(len(recorder2.record_x), 2 * (3 + 10 + 15))
        self.assertCollectionEqual(
                recorder1.record_x.data,
                recorder2.record_x[:len(recorder1.record_x)])

        # Test that time steps are contiguous and that there are no gaps.
        x = recorder2.record_x['micro'][:, 0, 0]
        dx = x[1:] - x[:-1]
        ddx = dx[1:] - dx[:-1]
        if np.any(ddx != 1):
            self.fail(f"wrong time steps:\n{x}\n{dx}\n{ddx}")

        # There should be 2 trajectories in the dataset, one for each run.
        num1 = len(led2.server.datasets.train_dataset.as_trajectories())
        num2 = len(led2.server.datasets.valid_dataset.as_trajectories())
        self.assertEqual(num1 + num2, 2)


class TestE2EPostprocessRunner(_TestE2ECase):
    def _test(self, backend=backends.backend_numpy.NumpyBackend, complex_state=False):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not found")

        _assertEqual = self.assertCollectionEqual if complex_state else self.assertArrayEqual

        # Set up.
        micro = _Micro(backend, complex_state)
        transformer = _Transformer(backend, complex_state)

        rnn = adaled.RNNConfig(input_size=8, output_size=8, append_F=False)
        macro = adaled.DeterministicPropagatorEnsemble(
                [adaled.RNN(rnn) for _ in range(5)])

        config = adaled.AdaLEDConfig(
                max_steps=150,
                quiet=True,
                dump_dataset=adaled.DumpConfig(every=0),
                dump_macro=adaled.DumpConfig(every=0),
                dump_transformer=adaled.DumpConfig(every=0),
                dump_diagnostics=adaled.DumpConfig(every=0))
        criteria = adaled.SimpleCriteriaConfig(
                k_warmup=3, k_cmp=8, max_micro_steps=100,
                max_macro_steps=5, num_relaxation_steps=0,
                max_cmp_error=1e10, max_uncertainty=1e10)  # Always accept.
        recorder = adaled.RecorderConfig(start_every=50, num_steps=50,
                                         path_fmt='record-{start_timestep:05d}.h5')

        # Run.
        datasets = adaled.utils.data.collections.CappedFixedLengthTrajectoryDatasetCollection(
            train_capacity=100, valid_capacity=10, trajectory_length=12)
        led = adaled.AdaLED(macro, config, criteria.create(), datasets,
                            transformer=transformer)
        led.run(led.make_generator(micro), recorder)

        # Load the initial. Check that x_micro is NaN in macro stages.
        record_paths = ['record-00000.h5', 'record-00050.h5', 'record-00100.h5']
        record_old = load_and_concat_records(record_paths)['fields']
        stages = record_old['metadata', 'stage']
        x_micro_old = record_old['simulations', 'x', 'micro']
        expected_stages = ([Stage.WARMUP] * 3 + [Stage.COMPARISON] * 8 + [Stage.MACRO] * 5) * 10
        expected_stages = [int(stage) for stage in expected_stages[:150]]
        self.assertArrayEqual(stages, expected_stages)
        _assertEqual(adaled.cmap(np.isnan, x_micro_old[stages == Stage.MACRO]),
                     adaled.cmap(lambda _: np.asarray(True), x_micro_old))
        for i in range(1, len(stages)):
            if stages[i] != Stage.MACRO and stages[i - 1] != Stage.MACRO:
                # This should be it, += i+1 is what the mock micro solver's advance does.
                _assertEqual(x_micro_old[i], x_micro_old[i - 1] + i + 1)

        # Postprocess run, fill NaN values.
        micro = _Micro(backend, complex_state)
        runner = PostprocessRunner(micro, recorder, record_paths)
        runner.run(verbose=False)

        # Check that NaN values are gone and that they have the correct value.
        record_new = load_and_concat_records(record_paths)
        x_micro_new = record_new['fields', 'simulations', 'x', 'micro']
        _assertEqual(adaled.cmap(np.isnan, x_micro_new),
                     adaled.cmap(lambda _: np.asarray(False), x_micro_new))
        _assertEqual(x_micro_new[stages != Stage.MACRO],
                     x_micro_old[stages != Stage.MACRO])
        for i in range(len(stages) - 1):
            if stages[i] == Stage.MACRO:
                # This should be it, += i+1 is what the mock micro solver's advance does.
                _assertEqual(x_micro_new[i], x_micro_new[i - 1] + i + 1)

    def test_simple_state_numpy(self):
        self._test(backend=backends.backend_numpy.NumpyBackend, complex_state=False)

    def test_complex_state_numpy(self):
        self._test(backend=backends.backend_numpy.NumpyBackend, complex_state=True)
