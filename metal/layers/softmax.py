import numpy as _np
from metal.layers.layer import Module

from metal.autograd import numpy as np
from metal.autograd import Container

class Softmax(Module):
    def __init__(self, dim=-1, optimizer=None):
        """
        A softmax nonlinearity layer.
        Notes
        -----
        This is implemented as a layer rather than an activation primarily
        because it requires retaining the layer input in order to compute the
        softmax gradients properly. In other words, in contrast to other
        simple activations, the softmax function and its gradient are not
        computed elementwise, and thus are more easily expressed as a layer.
        The softmax function computes:
        .. math::
            y_i = \\frac{e^{x_i}}{\sum_j e^{x_j}}
        where :math:`x_i` is the `i` th element of input example **x**.
        Parameters
        ----------
        dim: int
            The dimension in `X` along which the softmax will be computed.
            Default is -1.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None. Unused for this layer.
        """
        super().__init__(optimizer)

        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self):
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
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
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            Layer output for each of the `n_ex` examples.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)

        if retain_derived:
            pass

        return Y

    def _fwd(self, X):
        """Actual computation of softmax forward pass"""
        # center data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, retain_grads=True):
        pass
