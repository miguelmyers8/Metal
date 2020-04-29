import numpy as _np
from ..module.module import Module
from ..utils.function import softmax
from metal.autograd import numpy as np
from metal.autograd import Container

class Softmax(Module):
    def __init__(self, dim=-1, optimizer=None):
        super().__init__()
        self.dim = dim
        self.n_in = None
        self.is_initialized = False
        self.trainable = False
        self.updateable = False


    def _init_params(self):
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()
        return softmax(x, self.dim)

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
            "optimizer": {
                #"cache": self.optimizer.cache,
                #"hyperparameters": self.optimizer.hyperparameters,
            },
        }

Softmax.register()
