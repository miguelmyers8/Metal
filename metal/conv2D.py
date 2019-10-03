import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.dependency import Dependency
import math
import copy
from autograd.custom_function_operations import IMG2COL, Trans
from autograd.util import determine_padding, get_im2col_indices
from metal.layer import Layer


class Conv2D(Layer):
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1, seed=None):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True
        self.seed = seed

    def initialize(self, optimizer=None):
        np.random.seed(self.seed)
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.w = Parameter(data = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width)))
        self.b = Parameter(data = np.zeros((self.n_filters, 1)))
        # Weight optimizers
        if optimizer is not None:
            self.w_opt  = copy.copy(optimizer)
            self.b_opt = copy.copy(optimizer)

    def parameters_(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.INPUT = X
        # freezing the layer parameter if necessary
        if self.trainable == False:
            self.w.requires_grad = False
            self.b.requires_grad = False
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = IMG2COL(X, self.filter_shape, stride=self.stride, output_shape=self.padding).image_to_column()
        # Turn weights into column shape
        self.W_col = self.w.reshape((self.n_filters, -1))
        # Calculate output
        output = self.W_col @ self.X_col + self.b
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return Trans(output,axis_f=(3,0,1,2),axis_b=(1, 2, 3, 0)).trans()

    def backward_pass(self):
        if self.trainable:
            self.w = self.w_opt.update(self.w)
            self.b = self.b_opt.update(self.b)

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)
