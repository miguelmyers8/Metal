import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from metal.module import Module
from autograd.dependency import Dependency
import math
import copy
from metal.layers.layer import Layer

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

    def __init__(self, n_units, input_shape=None, seed=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.w = None
        self.b = None
        self.seed = seed

    def initialize(self, optimizer=None):
        np.random.seed(self.seed)
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.w = Parameter(data = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units)))
        self.b = Parameter(data = np.zeros((1, self.n_units)))
        # Weight optimizers
        if optimizer is not None:
            self.w_opt  = copy.copy(optimizer)
            self.b_opt = copy.copy(optimizer)

    def parameters_(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def forward_pass(self, X, training=True):
        assert (type(X) == Parameter) or (type(X) == Tensor), f"#{X} need to be Parameter or Tensor"
        self.layer_input = X
        # freezing the layer parameter if necessary
        if self.trainable == False:
            self.w.requires_grad = False
            self.b.requires_grad = False
        return X @ self.w + self.b

    def backward_pass(self):
        # Update the layer weights
        if self.trainable:
            self.w = self.w_opt.update(self.w)
            self.b = self.b_opt.update(self.b)
        # clear the gradients
        self.w.zero_grad()
        self.b.zero_grad()
        #self.X.zero_grad()

    def output_shape(self):
        return (self.n_units, )
