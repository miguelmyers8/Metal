import numpy as _np

from metal.autograd import numpy as np
from metal.autograd import Container

from metal.layers.layer import Module
from metal.initializers.activation_init import ActivationInitializer
from metal.initializers.weight_init import WeightInitializer
from metal.utils.utils import dtype


class Dense(Module):
    def __init__(self, n_out, act_fn=None, init="glorot_uniform", optimizer=None):
        """
        A fully-connected (dense) layer.
        Notes
        -----
        A fully connected layer computes the function
        .. math::
            \mathbf{Y} = f( \mathbf{WX} + \mathbf{b} )
        where `f` is the activation nonlinearity, **W** and **b** are
        parameters of the layer, and **X** is the minibatch of input examples.
        Parameters
        ----------
        n_out : int
            The dimensionality of the layer output
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The element-wise output nonlinearity used in computing `Y`. If None,
            use the identity function :math:`f(X) = X`. Default is None.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False


    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        W = Container(init_weights((self.n_in, self.n_out)),True)
        b = Container(np.zeros((1, self.n_out)),True)

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W":None, "b": None}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "FullyConnected",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
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
        if  retain_derived == False:
            W = self.parameters["W"]._value
            b = self.parameters["b"]._value
        else:
            W = self.parameters["W"]
            b = self.parameters["b"]

        Y, Z = self._fwd(X,W,b)

        if retain_derived:
            pass

        return Y

    def _fwd(self, X,W,b):
        """Actual computation of forward pass"""
        Z = X @ W + b
        Y = self.act_fn(Z)
        return Y, Z

    def backward(self,retain_grads=True):
        """
        Backprop from layer outputs to inputs.
        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            The gradient(s) of the loss wrt. the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
            The gradient of the loss wrt. the layer input(s) `X`.
        """
        assert self.trainable, "Layer is frozen"

        if retain_grads:
            self.gradients["W"] = self.parameters["W"].grad
            self.gradients["b"] = self.parameters["b"].grad
