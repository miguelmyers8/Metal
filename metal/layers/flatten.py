import numpy as _np
from ..module.module import Module
from ..core import FlattenFunction
from metal.autograd import numpy as np
from metal.autograd import Container

class Flatten(Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
        self.is_initialized = False

    def _init_params(self):
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self._init_params()
        return FlattenFunction(self.shape).forward(X)

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Flatten",
            "keep_dim": self.shape,
            "optimizer": {
                #"cache": self.optimizer.cache,
                #"hyperparameters": self.optimizer.hyperparameters,
            },
        }


Flatten.register()
