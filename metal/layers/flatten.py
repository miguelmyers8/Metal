import numpy as _np
from metal.layers.layer import Module
from metal.autograd import numpy as np
from metal.autograd import Container

class Flatten(Module):
    def __init__(self, keep_dim="first", optimizer=None):
        """
        Flatten a multidimensional input into a 2D matrix.
        Parameters
        ----------
        keep_dim : {'first', 'last', -1}
            The dimension of the original input to retain. Typically used for
            retaining the minibatch dimension.. If -1, flatten all dimensions.
            Default is 'first'.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """
        super().__init__(optimizer)

        self.keep_dim = keep_dim
        self.is_initialized = False


    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {"in_dims": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Flatten",
            "keep_dim": self.keep_dim,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>`
            Input volume to flatten.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(\*out_dims)`
            Flattened output. If `keep_dim` is `'first'`, `X` is reshaped to
            ``(X.shape[0], -1)``, otherwise ``(-1, X.shape[0])``.
        """
        if not self.is_initialized:
            self._init_params()

        if retain_derived:
            pass

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, retain_grads=True):
        pass
