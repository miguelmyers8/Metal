import numpy as _np

from metal.autograd import numpy as np
from metal.autograd import Container
from metal.initializers.activation_init import ActivationInitializer
from metal.initializers.weight_init import WeightInitializer
from metal.initializers.optimizer_init import OptimizerInitializer
from ..module.module import Module
from ..core import convolution_2d
from ..utils.utils import  _pair, parse_kwargs

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
        nobias=False, initialW=None, initial_bias=None,act_fn=None,
        optimizer=None,init="glorot_uniform", **kwargs):
        dilate, groups = parse_kwargs(kwargs, ('dilate', 1), ('groups', 1))
        super().__init__()
        self.init = init
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = int(groups)
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters_dict = {"W": None, "b": None}
        self.is_initialized = False


    def _init_params(self):
        kh, kw = _pair(self.ksize)
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        self.W = Container(init_weights((self.out_channels, int(self.in_channels / self.groups),kh,kw)),True)
        self.b = Container(np.zeros((1,1,1,self.out_channels)),True)
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self._init_params()
        return self.act_fn(convolution_2d(X, self.W, self.b, self.stride, self.pad, dilate=self.dilate, groups=self.groups, cudnn_fast=False))

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_channels,
            "out_ch": self.out_channels,
            "stride": self.stride,
            "dilation": self.dilate,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.ksize,
            "optimizer": {},
        }

Conv2D.register()
