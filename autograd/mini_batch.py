from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module
from typing import Sequence, Iterator, Tuple, Any
import numpy as np
import math

class mini_batch(object):
    """docstring for mini_batch."""
    def __init__(self, input_x: Tensor, input_y: Tensor, mini_batch_size: int, seed: int=1):
        super(mini_batch, self).__init__()
        self.input_x = input_x
        self.input_y = input_y
        self.mini_batch_size = mini_batch_size
        self.seed = seed


    def apply(self):

        np.random.seed(self.seed)

        self.mini_batches = []

        m = self.input_x.shape[1]

        permutation = list(np.random.permutation(m))

        self.input_x = self.input_x[:, permutation]
        self.input_y = self.input_y[:, permutation]

        num_mini_bath = math.floor(m/self.mini_batch_size)

        for k in range(0, num_mini_bath):

            mini_batch_X = self.input_x[:,k*self.mini_batch_size:(k+1)*self.mini_batch_size]
            mini_batch_Y = self.input_y[:,k*self.mini_batch_size:(k+1)*self.mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)

            self.mini_batches.append(mini_batch)

        if m % self.mini_batch_size != 0:
            mini_batch_X = self.input_x[:,m-self.mini_batch_size*num_mini_bath:m]
            mini_batch_Y = self.input_y[:,m-self.mini_batch_size*num_mini_bath:m]

            mini_batch = (mini_batch_X, mini_batch_Y)

            self.mini_batches.append(mini_batch)

        return self.mini_batches
