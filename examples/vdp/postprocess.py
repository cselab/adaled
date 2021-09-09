#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from examples.vdp.setup import Config, MicroSolver

from adaled import TensorCollection
from adaled.postprocessing.record import \
        CycleList, filter_and_slice_accepted_cycles, \
        get_rejected_cycles_last_cmp_step, load_and_concat_records
from adaled.utils.data.datasets import load_dataset
import adaled

import numpy as np

from typing import List
import glob

def error_func(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """The error function used in the report and the plots (currently the
    Euclidean distance)."""
    assert x1.shape == x2.shape and x1.shape[-1] == 2, (x1.shape, x2.shape)
    # return ((x1 - x2) ** 2).sum(axis=-1) / 2  # Internally in adaled.
    return ((x1 - x2) ** 2).sum(axis=-1) ** 0.5


def integrate_limit_cycle(
        mu: float,
        dt: float,
        integrator_dt: float,
        circular: bool) -> np.ndarray:
    """Integrate the VdP system and return the limit cycle."""
    print(f"Integrating mu={mu} dt={dt} integrator_dt={integrator_dt}...")

    # Low `mu` require multiple extra cycles to converge.
    num_cycles = 5 if mu < 0.6 else 3

    ics = np.array([[1.0, 0.0]])
    micro = MicroSolver(ics=ics, circular=circular,
                        dt_macro=dt, dt_micro=integrator_dt)

    cycle = 0
    record = adaled.DynamicArray()
    last = ics[0]
    step = 0
    while cycle < num_cycles:
        x = micro.advance(mu)[0].copy()
        step += 1
        if last[0] * x[0] < 0.0 and last[1] > 0.0 and x[1] > 0.0:
            if cycle == num_cycles - 2:
                record.append(last)
            cycle += 1
        if cycle == num_cycles - 1:
            record.append(x)
        last = x

    return record


def compute_dx_stats():
    """Compute stats of the residuals x_t - x_{t-1}, in order to check if
    network input/output scaling is needed."""
    d = load_dataset('dataset-latest/train')
    x = d.as_trajectories(('trajectory', 'x'))
    dx = x[:, 1:, :] - x[:, :-1, :]
    dx = np.abs(dx)
    print(dx.max(1))


def compute_accepted_cycles_info(cycles: CycleList):
    """
    Returns last_cmp and last_macro micro and macro steps of all accepted
    cycles. Here, last_macro macro stage is the expected state.
    """
    # If always_run_micro was not set to True, in principle we could rerun the
    # simulation starting from last_cmp step to compute the expected micro
    # state at the last_macro step.

    n = len(cycles.whole)
    xc_micro = np.empty((n, 2))
    xc_macro = np.empty((n, 2))
    xm_micro = np.empty((n, 2))
    xm_macro = np.empty((n, 2))
    c_uncertainty = np.empty(n)
    m_uncertainty = np.empty(n)
    comparison_errors = np.empty(n)
    mean_validation_errors = np.empty(n)   # Mean error in macro stage.
    final_validation_errors = np.empty(n)  # Final macro step error.
    start_timesteps = np.empty(n, dtype=np.int32)
    num_macro_steps = np.empty(n, dtype=np.int32)
    min_F = np.empty(n)
    max_F = np.empty(n)
    lengths = np.empty(n)
    for i, (whole, partial) in \
            enumerate(zip(cycles.whole, cycles.last_cmp_to_last_macro)):
        sim = partial['simulations']
        xc_micro[i] = sim['x', 'micro', 0]
        xc_macro[i] = sim['x', 'macro', 0]
        xm_micro[i] = sim['x', 'micro', -1]
        xm_macro[i] = sim['x', 'macro', -1]
        c_uncertainty[i] = sim['uncertainty', 0]
        m_uncertainty[i] = sim['uncertainty', -1]
        num_macro_steps[i] = len(sim) - 1
        comparison_errors[i] = error_func(xc_micro[i], xc_macro[i])
        mean_validation_errors[i] = error_func(sim['x', 'micro', 1:], sim['x', 'macro', 1:]).mean()
        final_validation_errors[i] = error_func(xm_micro[i], xm_macro[i])

        x = whole['simulations', 'x', 'micro']
        F = whole['simulations', 'F']
        start_timesteps[i] = whole['metadata', 'timestep', 0]
        lengths[i] = (((x[1:] - x[:-1]) ** 2).sum(axis=-1) ** 0.5).sum(axis=0)
        min_F[i] = F.min()
        max_F[i] = F.max()

    print(f"Found {n} accepted cycle(s).")
    info = TensorCollection({
        'last_cmp': {
            'x_micro': xc_micro,
            'x_macro': xc_macro,
            'uncertainty': c_uncertainty,
        },
        'last_macro': {
            'x_micro': xm_micro,
            'x_macro': xm_macro,
            'uncertainty': m_uncertainty,
        },
        'start_timestep': start_timesteps,
        'num_macro_steps': num_macro_steps,
        'comparison_error': comparison_errors,
        'mean_validation_error': mean_validation_errors,
        'final_validation_error': final_validation_errors,
        'length': lengths,
        'min_F': min_F,
        'max_F': max_F,
    })
    return info



def filter_bad_accepted_cycles(
        cycles: List[TensorCollection],
        distance_threshold: float = 0.3,
        F_threshold: float = 1.0):
    """Return a list of cycles (the whole trajectories) for which the macro
    state deviates from the expected micro for more than `distance_threshold`,
    with respect to the Euclidean distance."""

    bad = []
    errors = []
    for cycle in cycles:
        """Given a cycle batch, return those trajectories with error larger
        than the threshold."""
        assert len(cycle.keys()) == 2, cycle.keys()  # simulations and metadata
        sim = cycle['simulations']
        x_micro = sim['x', 'micro']
        length = (((x_micro[1:] - x_micro[:-1]) ** 2).sum(axis=-1) ** 0.5).sum(axis=0)
        x1_micro = sim['x', 'micro', -1]
        x1_macro = sim['x', 'macro', -1]
        distance = ((x1_macro - x1_micro) ** 2).sum(axis=-1) ** 0.5
        relative_error = distance / length
        F_ptp = sim['F'].ptp()
        for i, (dist, rel_error) in enumerate(zip(distance, relative_error)):
            if True or dist >= distance_threshold or F_ptp >= F_threshold:
                # Store distance such that we can sort from highest error to lowest.
                data = TensorCollection(
                        metadata=cycle['metadata'],
                        simulations=cycle['simulations', :, i])
                bad.append((float(dist), float(rel_error), F_ptp, data))

    bad.sort(key=lambda x: -(x[0] + x[2]))
    print("    dist  rel_error   F_ptp     timesteps")
    for dist, rel_error, F_ptp, cycle in bad[:50]:
        ts = cycle['metadata', 'timestep']
        print(f"{dist:8.4f}  {rel_error:9.4f}  {F_ptp:8.4f}   {ts[0]:6d}..{ts[-1]}")
    bad = [x[-1] for x in bad]

    # print(f"Found {len(bad)} bad accepted cycle(s), with Euclidean error "
    #       f"larger than {distance_threshold} or F_ptp larger than {F_threshold}.")
    print(f"Found {len(bad)} accepted cycle(s).")
    return bad


def find_well_defined_limit_cycles(
        config: Config,
        cycles: List[TensorCollection],
        mu_epsilon: float = 1e-5):
    """Find cycles with fixed `mu` and compute the corresponding limit cycles.

    Returns a dictionary `{<mu>: {'micro': <...>, 'macro': <...>}}`, where
    `micro` and `macro` denote the same cycle sampled with time step of
    `dt_micro` and `dt_rnn`, respectively.

    In the case of brownian motion, likely no fixed `mu` will be found, in
    which case an empty dictionary is returned.
    """
    mus = [1e9]  # Add dummy value to simplify grouping.
    for cycle in cycles:
        mu = cycle['simulations', 'F']
        # Is mu not constant, can its limit cycle be properly visualized?
        if np.ptp(mu) < mu_epsilon:  # max - min
            mus.append(mu.mean())
    mus.sort()

    groups = {}
    last_mu = mus[0]
    for mu in mus:
        if mu - last_mu >= mu_epsilon:
            print(f"Found mu group: mu={last_mu}")
            circular = config.circular_motion_system
            groups[last_mu] = {
                'micro': integrate_limit_cycle(
                        last_mu, config.dt_micro, config.dt_micro, circular),
                'macro': integrate_limit_cycle(
                        last_mu, config.dt_rnn, config.dt_micro, circular)[:-2],
            }
        last_mu = mu

    return groups


def postprocess_macro_cycles(config: Config):
    config: Config = adaled.load('config.pt')
    paths = sorted(glob.glob('record-0*.pt'))
    record = load_and_concat_records(paths)
    if len(record.keys()) == 0:
        return {}

    accepted_cycles = filter_and_slice_accepted_cycles(record)
    rejected_last_cmp = get_rejected_cycles_last_cmp_step(record)
    print(f"Found {len(rejected_last_cmp)} rejected cycles.")
    if not config.led.always_run_micro:
        raise NotImplementedError("not yet implemented")

    accepted_info = compute_accepted_cycles_info(accepted_cycles)
    bad = filter_bad_accepted_cycles(accepted_cycles.whole)
    limit_cycles = find_well_defined_limit_cycles(config, accepted_cycles.whole)

    output = {
        'version': 3,
        'accepted_cycles_info': accepted_info,
        'rejected_last_cmp': rejected_last_cmp,
        'bad_accepted_cycles': bad,
        'limit_cycles': limit_cycles,
    }
    return output


def main():
    # compute_dx_stats()
    config = adaled.load('config.pt')
    groups = postprocess_macro_cycles(config)
    adaled.save(groups, 'postprocessed-macro-cycles.pt')


if __name__ == '__main__':
    main()
