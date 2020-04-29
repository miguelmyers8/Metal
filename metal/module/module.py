import numpy as _np
from abc import ABC, abstractmethod
from ..initializers.optimizer_init import OptimizerInitializer
from ..initializers.activation_init import ActivationInitializer
from ..autograd import Container, Container_, is_container
import inspect


# =============================================================================
#  (base class)
# =============================================================================

class Module(ABC):
    types = set()

    def __init__(self, optimizer=None):
        """An abstract base class inherited by all layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.gradients = {}
        self.parameters_dict = {}
        self.derived_variables = {}
        self._modules = {}
        self._params = set()
        super().__init__()

    def __setattr__(self, name, value):
        if isinstance(value, (Container_, Module)):
            self._params.add(name)
        super().__setattr__(name, value)

    def _init_layer(self, **kwargs):
        pass

    def forward(self, z, **kwargs):
        pass

    def backward(self, **kwargs):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass

    def flush_gradients(self):
        pass

    def set_params(self, summary_dict):
        pass

    def summary(self):
        pass
    def _flatten_params(self, params_dict, parent_key=""):
        pass

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    # use numpy
    def to_cpu(self):
        pass
    # use cupy
    def to_gpu(self):
        pass

    # returns modules and name
    def named_modules(self):
        for name in self._params:
            obj = self.__dict__[name]
            if is_module(obj):
                yield name, obj

    # returns modules
    def modules(self):
        for name, mod in self.named_modules():
            yield  mod

    # set layer to training
    def train(self, mode=True):
        self.training = mode
        for obj in self.modules():
            obj.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # return parameters from modules
    def parameters(self):
        if self._params:
            for name in self._params:
                obj = self.__dict__[name]
                if is_module(obj):
                    yield from obj.parameters()
                else:
                    yield obj

    # clears the gradients
    def zero_grad(self):
        for containers in self.parameters():
            containers.zero_grad()

    # TODO: needs to be faster we have some float64 coming from autograd that slowing use down
    # update a layers parameters
    def update(self, cur_loss=None):
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for p,n in self.parameters():
            p._value = self.optimizer(p._value, p.grad, n, cur_loss)
            p.zero_grad()

    # register subclasses so we can find them. faster than isinstance
    @classmethod
    def register(cls):
        Module.types.add(cls)

_module_types = Module.types
is_module  = lambda x: type(x) in _module_types
