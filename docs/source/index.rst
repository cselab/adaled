Adaptive Learning of Effective Dynamics (AdaLED)
================================================


Adaptive Learning of Effective Dynamics (AdaLED) is a framework for building online parametric surrogate models based on machine learning.
The framework runs a given simulation, records its trajectory and on the fly trains the surrogate to reproduce the simulation dynamics.
The framework constantly monitors the surrogate's prediction accuracy and uncertainty.
Whenever the accuracy is sufficiently high, the framework transitions the simulation control from the original simulation to the surrogate.
To ensure that the surrogate does not deviate greatly from the target trajectory, the control is transferred back once the prediction uncertainty crosses a given threshold.

AdaLED is suitable for systems that are expensive to simulate but exhibit a lower-order effective dynamics.
Thus, two scales are differentiated: the *micro scale* (the original detailed state space that the simulation operates on) and the *macro scale* (the lower-order state space).
The surrogate is composed of two parts: the autoencoder and the macro propagator.
The autoencoder identifies the macro representation and transforms the micro state to the macro state and back.
The macro propagator is repsonsible for learning the macro dynamics and advancing the simulation state in the macro scale.



.. toctree::
    :maxdepth: 2
    :caption: Contents:

    dataset

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
