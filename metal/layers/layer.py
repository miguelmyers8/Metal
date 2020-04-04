import numpy as _np
from abc import ABC, abstractmethod
from metal.initializers.optimizer_init import OptimizerInitializer
from metal.initializers.activation_init import ActivationInitializer
from metal.autograd import Container
import inspect


# =============================================================================
# Layer (base class)
# =============================================================================

class Module(ABC):
    def __init__(self, optimizer=None):
        """An abstract base class inherited by all nerual network layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = OptimizerInitializer(optimizer)()

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, **kwargs):
        raise NotImplementedError

    def change_optimizer(self,optimizer=None):
        self.optimizer = OptimizerInitializer(optimizer)()

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"
        self.X = []

        #for k, v in self.parameters.items():
            #self.parameters[k].grad = None

        for k, v in self.gradients.items():
            self.gradients[k] = None

    def update(self, cur_loss=None):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = Container(self.optimizer(self.parameters[k]._value, v, k, cur_loss),True)
        self.flush_gradients()

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.
        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.
        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        layer, sd = self, summary_dict

        # collapse `parameters` and `hyperparameters` nested dicts into a single
        # merged dictionary
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        for k, v in sd.items():
            if k in self.parameters:
                layer.parameters[k] = v
            if k in self.hyperparameters:
                if k == "act_fn":
                    layer.act_fn = ActivationInitializer(v)()
                if k == "optimizer":
                    layer.optimizer = OptimizerInitializer(sd[k])()
                if k not in ["wrappers", "optimizer"]:
                    setattr(layer, k, v)
                if k == "wrappers":
                    layer = init_wrappers(layer, sd[k])
        return layer

    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID."""
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }


    def _flatten_params(self, params_dict, parent_key=""):
        c=-1
        for name, value in inspect.getmembers(self):
            if isinstance(value, Module):
                c+=1
                key = parent_key + '/' + value.__class__.__name__ + str(c) if parent_key else value.__class__.__name__ + str(c)
                value._flatten_params(params_dict, key)

        for params_name, params_value in self.parameters.items():
            key = parent_key + '/' + params_name if parent_key else params_name
            if params_value.grad is not None:
                params_value.cleargrad()
            params_dict[key] = params_value

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
            if params_value.grad is not None:
                params.cleargrad()
            param._value = npz[key]
