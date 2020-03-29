from metal.layers.layer import LayerBase
from metal.utils.utils import pad2D
from metal.utils.functional import _pool2D

class Pool2D(LayerBase):
    def __init__(self, kernel_shape, stride=1, pad=0, mode="max", optimizer=None, data_format="channels_last"):
        """
        A single two-dimensional pooling layer.
        Parameters
        ----------
        kernel_shape : 2-tuple
            The dimension of a single 2D filter/kernel in the current layer
        stride : int
            The stride/hop of the convolution kernels as they move over the
            input volume. Default is 1.
        pad : int, tuple, or 'same'
            The number of rows/columns of 0's to pad the input. Default is 0.
        mode : {"max", "average"}
            The pooling function to apply.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """
        super().__init__(optimizer)

        self.pad = pad
        self.mode = mode
        self.in_ch = None
        self.out_ch = None
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.is_initialized = False
        self.data_format = data_format

    def _init_params(self):
        self.derived_variables = {"out_rows": [], "out_cols": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Pool2D",
            "act_fn": None,
            "pad": self.pad,
            "mode": self.mode,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
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
            (`in_rows`,`in_cols`, `in_ch`)
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
            self.in_ch = self.out_ch = X.shape[3]
            self._init_params()

        Y = _pool2D(X, self.kernel_shape, self.stride, self.pad,self.mode, self.data_format)
        return Y

    def backward(self, retain_grads=True):
        pass
