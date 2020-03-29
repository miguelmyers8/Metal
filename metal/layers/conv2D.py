import numpy as _np


from metal.autograd import numpy as np
from metal.autograd import Container

from metal.layers.layer import LayerBase
from metal.initializers.activation_init import ActivationInitializer
from metal.initializers.weight_init import WeightInitializer
from metal.utils.utils import pad2D, conv2D, im2col, col2im, dilate, calc_pad_dims_2D, dtype

class Conv2D(LayerBase):
    def __init__(self,out_ch,kernel_shape,pad=0,stride=1,dilation=0,act_fn=None,optimizer=None,init="glorot_uniform"):
        """
        Apply a two-dimensional convolution kernel over an input volume.
        Notes
        -----
        Equations::
            out = act_fn(pad(X) * W + b)
            n_rows_out = floor(1 + (n_rows_in + pad_left + pad_right - filter_rows) / stride)
            n_cols_out = floor(1 + (n_cols_in + pad_top + pad_bottom - filter_cols) / stride)
        where `'*'` denotes the cross-correlation operation with stride `s` and
        dilation `d`.
        Parameters
        ----------
        out_ch : int
            The number of filters/kernels to compute in the current layer
        kernel_shape : 2-tuple
            The dimension of a single 2D filter/kernel in the current layer
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``Y[t]``. If None, use the
            identity function :math:`f(X) = X` by default. Default is None.
        pad : int, tuple, or 'same'
            The number of rows/columns to zero-pad the input with. Default is
            0.
        stride : int
            The stride/hop of the convolution kernels as they move over the
            input volume. Default is 1.
        dilation : int
            Number of pixels inserted between kernel elements. Effective kernel
            shape after dilation is: ``[kernel_rows * (d + 1) - d, kernel_cols
            * (d + 1) - d]``. Default is 0.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """
        super().__init__(optimizer)

        self.pad = pad
        self.init = init
        self.in_ch = None
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.kernel_shape = kernel_shape
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        fr, fc = self.kernel_shape
        W = Container(init_weights((fr, fc, self.in_ch, self.out_ch)),True)
        b = Container(np.zeros((1, 1, 1, self.out_ch)),True)

        self.parameters = {"W": W, "b": b}
        self.gradients = {"W": None, "b": None}
        self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output given input volume `X`.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The layer output.
        """
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()
        if  retain_derived == False:
            W = self.parameters["W"]._value
            b = self.parameters["b"]._value
        else:
            W = self.parameters["W"]
            b = self.parameters["b"]

        n_ex, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        # pad the input and perform the forward convolution
        Z = conv2D(X, W, s, p, d) + b
        Y = self.act_fn(Z)

        if retain_derived:
            pass

        return Y

    def backward(self, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.
        Notes
        -----
        Relies on :meth:`~numpy_ml.neural_nets.utils.im2col` and
        :meth:`~numpy_ml.neural_nets.utils.col2im` to vectorize the
        gradient calculation.
        See the private method :meth:`_backward_naive` for a more straightforward
        implementation.
        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows,
        out_cols, out_ch)` or list of arrays
            The gradient(s) of the loss with respect to the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss with respect to the layer input volume.
        """
        assert self.trainable, "Layer is frozen"

        if retain_grads:
            self.gradients["W"] = self.parameters["W"].grad
            self.gradients["b"] = self.parameters["b"].grad
