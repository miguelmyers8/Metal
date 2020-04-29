import numpy as _np
from ..module.module import Module
from ..utils.function import flatten
from metal.autograd import numpy as np
from metal.autograd import Container

class Flatten(Module):
    def __init__(self, keep_dim="first", optimizer=None):
        super().__init__()
        self.keep_dim = keep_dim
        self.is_initialized = False
        self.trainable = False
        self.updateable = False


    def _init_params(self):
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self._init_params()
        return flatten(X,self.keep_dim)

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Flatten",
            "keep_dim": self.keep_dim,
            "optimizer": {
                #"cache": self.optimizer.cache,
                #"hyperparameters": self.optimizer.hyperparameters,
            },
        }


Flatten.register()
