import numpy as _np

from metal.autograd import numpy as np
from metal.autograd import Container
from metal.initializers.activation_init import ActivationInitializer
from metal.initializers.weight_init import WeightInitializer
from metal.initializers.optimizer_init import OptimizerInitializer

from ..module.module import Module
from ..utils.function import pad2D, conv2D, dilate

class Conv2D(Module):
    def __init__(self,out_ch,kernel_shape,pad=0,stride=1,dilation=0,act_fn=None,optimizer=None,init="glorot_uniform"):
        super().__init__()
        self.pad = pad
        self.init = init
        self.in_ch = None
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.kernel_shape = kernel_shape
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters_dict = {"W": None, "b": None}
        self.is_initialized = False


    def _init_layer(self):
        fr, fc = self.kernel_shape
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        self.W = Container(init_weights((fr, fc, self.in_ch, self.out_ch)),True)
        self.b = Container(np.zeros((1, 1, 1, self.out_ch)),True)
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()
        n_ex, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation
        return self.act_fn((conv2D(X, self.W, s, p, d) + self.b))

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

            },
        }

Conv2D.register()
