Basic autoencoders
==================

Basic
-----

.. autoclass:: adaled.IdentityTransformer
    :members:

.. autoclass:: adaled.AutoencoderTransformer
    :members:

.. autoclass:: adaled.StackedTransformer
    :members:


Scaling
-------

.. note::
    There are currently two autoencoders/transformers for scaling, the :class:`ScalingTransformer` and :class:`ScaledAutoencoderTransformer`.
    The latter is more general and allows for custom :attr:`inv_scaling`, which can be useful when the output of the autoencoder is not entirely compatible with the input.

.. autoclass:: adaled.transformers.scaling.Scaling

.. autoclass:: adaled.transformers.scaling.ScalingLayer

.. autoclass:: adaled.transformers.scaling.ScalingTransformer
