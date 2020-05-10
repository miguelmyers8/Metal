import numpy as _np
from metal.autograd import numpy as np
from metal.autograd import Container
from metal.initializers.activation_init import ActivationInitializer
from metal.initializers.weight_init import WeightInitializer
from metal.initializers.optimizer_init import OptimizerInitializer
from ..module.module import Module
from ..core import dense

class Dense(Module):
    def __init__(self, n_in, n_out, act_fn=None, init="glorot_uniform", optimizer=None):
        super().__init__()
        self.init = init
        self.n_in = n_in
        self.n_out = n_out
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters_dict = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        self.W = Container(init_weights((self.n_out, self.n_in)),True)
        self.b = Container(np.zeros((1, self.n_out)),True)
        self.is_initialized = True


    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self._init_params()
        return  self.act_fn(dense(X,self.W,self.b))


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

            },
        }

Dense.register()
