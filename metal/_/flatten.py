import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from metal.module import Module
from autograd.dependency import Dependency
import math
import copy
from metal.layers.layer import Layer

class Flatten(Layer):
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self):
        # has no grad or update
        pass

    def output_shape(self):
        return (np.prod(self.input_shape),)
