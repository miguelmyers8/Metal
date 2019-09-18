import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from metal.module import Module
from autograd.dependency import Dependency
import math
import copy

class Layer(Module):
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

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()

class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    example shape:
        w = np.random.randn(6,5)
        i = np.random.randn(2,6)
        np.dot(i,w) + np.random.randn(1,5)
    """

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.w = None
        self.b = None

    def initialize(self, optimizer=None):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.w = Parameter(inputs_ =np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units)))
        self.b = Parameter(inputs_ =np.zeros((1, self.n_units)))
        # Weight optimizers
        if optimizer is not None:
            self.w_opt  = copy.copy(optimizer)
            self.b_opt = copy.copy(optimizer)

    def parameters_(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def forward_pass(self, inputs, training=True):
        assert (type(inputs) == Parameter) or (type(inputs) == Tensor), f"#{inputs} need to be Parameter or Tensor"
        # freezing the layer parameter if necessary
        if self.trainable == False:
            self.w.requires_grad = False
            self.b.requires_grad = False
        return inputs @ self.w + self.b

    def backward_pass(self):
        # Update the layer weights
        self.w = self.w_opt.update(self.w)
        self.b = self.b_opt.update(self.b)

        self.w.zero_grad()
        self.b.zero_grad()
        self.inputs.zero_grad()

    def output_shape(self):
        return (self.n_units, )
