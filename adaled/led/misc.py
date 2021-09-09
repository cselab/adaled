import adaled

import os

PLOT_SCRIPT_TEMPLATE = '''\
#!/bin/bash

# Use single thread because plotting scripts internally uses multiprocessing.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {adaled_path}/plotting/all.py "$@"
'''

def init_output_folder(dir: str, verbose: bool = False):
    """Create the output folder and add a plotting shortcut script."""
    import shlex
    adaled_path = os.path.dirname(os.path.abspath(adaled.__file__))
    script = PLOT_SCRIPT_TEMPLATE.format(
            adaled_path=shlex.quote(adaled_path))
    adaled.save_executable_script(
            os.path.join(dir, 'plot.sh'), script, verbose=verbose)
