from autograd.module import Module
from autograd.tensor import Tensor, Dependency
from autograd.linear import Linear
from autograd.nn import Sequential

from functools import reduce
import numpy as np

class Grad_Check(Module):
    """docstring for Grad_Check."""

    def __init__(self):
        super(Grad_Check, self).__init__()


    def dictionary_to_vector(self):

        keys = []
        count = 0

        for key in self.param:
            new_vector = np.reshape(self.param[key].data, (-1,1))
            keys = keys + [key]*new_vector.shape[0]
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys

    def vector_to_dictionary(self, th=None):

        theta = self.param
        theta_,_ = self.dictionary_to_vector()
        self.dic = dict()
        count = 0

        if th is not None:
            theta_ = th

        #take incoming vector and reshape it back to orginal.
        for i, (key, val) in enumerate(theta.items()):
            if i == 0:
                data_shape = reduce(lambda x, y: x*y, theta[key].shape)
                self.dic[key] = theta_[count:data_shape].reshape(theta[key].shape)
                count = data_shape
            else:
                data_shape = reduce(lambda x, y: x*y, theta[key].shape)
                s = (count+data_shape)
                self.dic[key] = theta_[count:s].reshape(theta[key].shape)
                count = s
        return self.dic

    def gradients_to_vector(self):

        count = 0
        for key, _ in self.grad.items():
            
            # flatten parameter
            new_vector = np.reshape(self.grad[key].data, (-1,1))

            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta
