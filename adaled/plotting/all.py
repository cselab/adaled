#!/usr/bin/env python3

"""Run all plotting scripts with the latest data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adaled.plotting.tasks import discover_and_run
import adaled.plotting.plot_dataset as plot_dataset
import adaled.plotting.plot_diagnostics as plot_diagnostics
import adaled.plotting.plot_macro as plot_macro
import adaled.plotting.plot_postprocessed as plot_postprocessed
import adaled.plotting.plot_record as plot_record
import adaled.plotting.plot_transformer as plot_transformer

import glob

def get_default_plotter_classes():
    # TODO: Replace this file with automatic discovery, similar to what unittest has.
    # TODO: Plotter should be able to automatically skip themselves if the
    # required files are not found.
    plotter_classes = [
        plot_dataset.DatasetPlotter,
        plot_record.PerRecordTrajectoryPlotter,
        plot_record.MergedRecordsPlotter,
    ]

    # diagnostics.pkl is legacy (2021-11-19).
    if os.path.exists('diagnostics.pkl') or glob.glob('diagnostics-*.pt'):
        plotter_classes.append(plot_diagnostics.DiagnosticsPlotter)
    else:
        print("Skipping diagnostics plots, diagnostics dump not found.")

    if os.path.exists('transformer-latest.pt'):
        plotter_classes.append(plot_transformer.TransformerPlotter)
        plotter_classes.append(plot_macro.MacroTrajectoryPlotter)
    else:
        print("Skipping transformer plots, transformer dump not found.")

    if os.path.exists('postprocessed_data_1d.pt'):
        plotter_classes.append(plot_postprocessed.PlotPostprocessed1D)
    else:
        print("Postprocessed multi-channel 1D data not found, skipping.")

    return plotter_classes


def main(extra_classes=[]):
    """Run multiple plotters, including the default ones.

    In case the user-provided classes override some of the default ones, the
    base default classes will be skipped.
    """
    default = get_default_plotter_classes()
    default = [cls for cls in default
               if not any(issubclass(ex, cls) for ex in extra_classes)]
    discover_and_run(default + extra_classes)


if __name__ == '__main__':
    main()
