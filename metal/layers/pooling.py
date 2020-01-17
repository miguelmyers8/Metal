import numpy as np
from metal.layers.layer import LayerBase
from metal.utils.utils import pad2D

class Pool2D(LayerBase):
    def __init__(self, kernel_shape, stride=1, pad=0, mode="max", optimizer=None):
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

    def __call__(self, optimizer):
        super().__init__(optimizer)
        return self
        
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

        n_ex, in_rows, in_cols, nc_in = X.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

        out_rows = np.floor(1 + (in_rows + pr1 + pr2 - fr) / s).astype(int)
        out_cols = np.floor(1 + (in_cols + pc1 + pc2 - fc) / s).astype(int)

        if self.mode == "max":
            pool_fn = np.max
        elif self.mode == "average":
            pool_fn = np.mean

        Y = np.zeros((n_ex, out_rows, out_cols, self.out_ch))
        for m in range(n_ex):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(self.out_ch):
                        # calculate window boundaries, incorporating stride
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        xi = X_pad[m, i0:i1, j0:j1, c]
                        Y[m, i, j, c] = pool_fn(xi)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["out_rows"].append(out_rows)
            self.derived_variables["out_cols"].append(out_cols)

        return Y



    def backward(self, dLdY, retain_grads=True):
        """
        Backprop from layer outputs to inputs
        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer input `X`.
        """
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        Xs = self.X
        out_rows = self.derived_variables["out_rows"]
        out_cols = self.derived_variables["out_cols"]

        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad

        dXs = []
        for X, dy, out_row, out_col in zip(Xs, dLdY, out_rows, out_cols):
            n_ex, in_rows, in_cols, nc_in = X.shape
            X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

            dX = np.zeros_like(X_pad)
            for m in range(n_ex):
                for i in range(out_row):
                    for j in range(out_col):
                        for c in range(self.out_ch):
                            # calculate window boundaries, incorporating stride
                            i0, i1 = i * s, (i * s) + fr
                            j0, j1 = j * s, (j * s) + fc

                            if self.mode == "max":
                                xi = X[m, i0:i1, j0:j1, c]

                                # enforce that the mask can only consist of a
                                # single `True` entry, even if multiple entries in
                                # xi are equal to max(xi)
                                mask = np.zeros_like(xi).astype(bool)
                                x, y = np.argwhere(xi == np.max(xi))[0]
                                mask[x, y] = True

                                dX[m, i0:i1, j0:j1, c] += mask * dy[m, i, j, c]
                            elif self.mode == "average":
                                frame = np.ones((fr, fc)) * dy[m, i, j, c]
                                dX[m, i0:i1, j0:j1, c] += frame / np.prod((fr, fc))

            pr2 = None if pr2 == 0 else -pr2
            pc2 = None if pc2 == 0 else -pc2
            dXs.append(dX[:, pr1:pr2, pc1:pc2, :])
        return dXs[0] if len(Xs) == 1 else dXs
