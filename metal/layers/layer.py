import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from metal.module import Module
from autograd.dependency import Dependency
import math
import copy
from metal.layers.activation_functions import Sigmoid, ReLU, TanH, Softmax



activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': TanH,
    'softmax':Softmax,
}

class Layer(Module):
    __slots__ = ('input_shape')
    """docstring for Layer."""

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters_(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def update_pass(self):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()



class Activation(Layer):
    __slots__=('activation_name','activation_func','trainable','layer_input')
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def update_pass(self):
        self.zero_grad()

    def output_shape(self):
        return self.input_shape
