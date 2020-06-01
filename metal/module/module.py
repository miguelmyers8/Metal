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
        self.training = True
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

    def __call__(self, x):
        return self.forward(x)

    def _init_params(self, **kwargs):
        for i in self.modules():
            i._init_params()

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

    # use numpy
    def to_cpu(self):
        pass
    # use cupy
    def to_gpu(self):
        pass

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if is_module(obj):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param._value for key, param in params_dict.items()
                      if param is not None}
        try:
            _np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = _np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param._value = npz[key]

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

    # return name and parameters from modules
    def name_parameters(self):
        if self._params:
            for name in self._params:
                obj = self.__dict__[name]
                if is_module(obj):
                    yield from obj.name_parameters()
                else:
                    yield name, obj


    # return parameters from modules
    def parameters(self):
        for name, parms in self.name_parameters():
            yield  parms


    # clears the gradients
    def zero_grad(self):
        for containers in self.parameters():
            containers.zero_grad()

    # TODO: needs to be faster we have some float64 coming from autograd that slowing use down
    # update a layers parameters
    def update(self, cur_loss=None):
        assert self.training, "Layer is frozen"
        self.optimizer.step()
        for p, n in self.name_parameters():
            p._value = self.optimizer(p._value, p.grad, n, cur_loss)
            p.zero_grad()

    # register subclasses so we can find them. faster than isinstance
    @classmethod
    def register(cls):
        Module.types.add(cls)

_module_types = Module.types
is_module  = lambda x: type(x) in _module_types
is_module_list = lambda l: list(map(is_module,l))
