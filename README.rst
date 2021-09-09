AdaLED: Adaptive Learning of Effective Dynamics
===============================================

AdaLED is a method for accelerating computationally expensive simulations by alternating between the slow original simulator and a fast machine-learning-based online surrogate model.

The surrogate model is trained on the fly on the trajectories produced by the original simulator.
It uses an autoencoder to discover a low-dimensional latent representation of the system states and an ensemble of recurrent neural networks (RNNs) to learn the trajectories, i.e., the system dynamics.

Moreover, AdaLED continuously monitors the prediction error of the surrogate model.
AdaLED replaces the original simulator with the surrogate only when the prediction error is below a user-specified threshold.
The ensemble of RNNS not only predicts the dynamics but also estimates the confidence of the prediction.
Depending on the confidence, AdaLED will utilize the surrogate for more or fewer time steps before transferring the simulation back from the original simulator.

More information about the method can be found in the paper [1]_.


Installation
------------

.. code-block:: bash

    mkdir adaled
    cd adaled
    python3 -m venv venv
    echo "export PYTHONPATH=$(pwd)/adaled:\$PYTHONPATH" >> venv/bin/activate
    source venv/bin/activate

    git clone https://github.com/cselab/adaled
    cd adaled

    python3 -m pip install -r requirements.txt

    # Run unit and end to end tests.
    cd tests
    ./run.sh -f
    ./e2e_examples.py -f


The code was tested with Python 3.9, but should work with newer versions as well.


Examples
--------

Three examples are included in the repository:

1. Van der Pol (VdP) oscillator: a 2-variable system, periodic, no autoencoder

2. Reaction-diffusion equation: a 2D partial differential equation with periodic dynamics

3. 2D flow behind the cylinder: a 2D partial differential equation with approx. periodic dynamics

Each example contains a system parameter that determines the dynamics and that changes periodically throughout the simulation.
When the parameter changes to a new value, AdaLED detects that the system entered an unseen regime and turns off the surrogate until it learns this new dynamics.


To run the examples, go to the corresponding folder and run ``run.py``.

.. code-block:: bash

    cd examples/vdp/
    ./run.py


The output of AdaLED, which includes the simulation trajectories and various diagnostics, is stored in the folder ``output``.
To plot the results (during or after the run), run the following:

.. code-block:: bash

    cd output
    ./postprocess.sh
    ./plot.sh

    # To generate an animation.
    ./movie.sh -j4


In the VdP example, it takes about 10-15 mins to learn the dynamics.
For the reaction-diffusion and flow behind the cylinder, it takes a few hours.


2D flow behind the cylinder
...........................

The 2D flow behind the cylinder example uses the `CubismAMR <https://github.com/cselab/CUP2D>`_ solver.
To install the solver, clone it, run ``git submodule update --init`` from within the repository, and follow the cmake compilation instructions.
Please note that the solver requires HDF5 and MPI.


Troubleshooting
---------------

If an error *h5py was built without MPI support, can't use mpio driver* appears, reinstall ``h5py`` using the following commands:

.. code-block:: bash

    CC=mpicc HDF5_MPI=ON pip install --force-reinstall --no-binary=h5py h5py

You might need to use ``CC=cc`` instead of ``CC=mpicc``.

When using CUDA, the plotting script will not be able to use parallelization by itself.
Instead, use one of the following:

.. code-block:: bash

    # Disable CUDA to run the parallelized plotting script.
    CUDA_AVAILABLE_DEVICES= ./plot.sh

    # Or use MPI to parallelize. mpi4py must be available.
    mpirun -n 8 ./plot.sh -j1

Publications
------------

.. [1] *Adaptive learning of effective dynamics for online modeling of complex systems*, I. Kičić, P. Vlachas, G. Arampatzis, M. Chatzimanolakis, L. Guibas, P. Koumoutsakos
.. [2] *Multiscale Simulations of Complex Systems by Learning their Effective Dynamics*,Vlachas, G. Arampatzis, C. Uhler, P. Koumoutsakos, Nat. Mach. Intell., 2022.
.. [3] *Accelerated Simulations of Molecular Systems through Learning of their Effective Dynamics*, PR. Vlachas, J. Zavadlav, M. Praprotnik, P. Koumoutsakos, J. Chem. Theory Comput., vol. 18, iss. 1, pp. 538-549, 2021.
.. [4] *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, Pantelis R. Vlachas, Wonmin Byeon, Zhong Y. Wan, Themistoklis P. Sapsis and Petros Koumoutsakos, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018.
