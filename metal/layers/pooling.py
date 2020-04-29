from ..module.module import Module
from ..utils.function import _pool2D
from ..utils.functional import pad2D


class Pool2D(Module):
    def __init__(self, kernel_shape, stride=1, pad=0, mode="max", optimizer=None, data_format="channels_last"):
        super().__init__()
        self.pad = pad
        self.mode = mode
        self.in_ch = None
        self.out_ch = None
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.is_initialized = False
        self.data_format = data_format
        self.parameters_dict = {}

    def _init_params(self):
        self.derived_variables = {"out_rows": [], "out_cols": []}
        self.is_initialized = True


    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.in_ch = self.out_ch = X.shape[3]
            self._init_params()
        return _pool2D(X, self.kernel_shape, self.stride, self.pad,self.mode, self.data_format)


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
                #"cache": self.optimizer.cache,
                #"hyperparameters": self.optimizer.hyperparameters,
            },
        }
Pool2D.register()
