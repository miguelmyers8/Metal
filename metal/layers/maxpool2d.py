from ..module.module import Module
from ..core import max_pooling_2d


class MaxPool2D(Module):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
        super().__init__()
        self.pad = pad
        self.stride = stride
        self.ksize = ksize
        self.is_initialized = False
        self.cover_all = cover_all
        self.return_indices = return_indices
        self.parameters_dict = {}

    def _init_params(self):
        self.derived_variables = {"out_rows": [], "out_cols": []}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self._init_params()
        return max_pooling_2d(X, self.ksize, self.stride, self.pad, self.cover_all, self.return_indices)

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Pool2D",
            "act_fn": None,
            "pad": self.pad,
            "stride": self.stride,
            "kernel_shape": self.ksize,
            "optimizer": {
                #"cache": self.optimizer.cache,
                #"hyperparameters": self.optimizer.hyperparameters,
            },
        }

MaxPool2D.register()
