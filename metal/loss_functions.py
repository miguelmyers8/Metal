from __future__ import division
import numpy as np
from metal.utils.data_operation import accuracy_score
from metal.layers.activation_functions import Sigmoid
from autograd.dependency import Dependency

class Loss(object):
    def loss(self, y, p):
        return NotImplementedError()

    def gradient(self, y, p):
        raise NotImplementedError()

    def acc(self, y, p):
        return 0

class SquareLoss(Loss):
    def __init__(self,y=None,p=None):
        self.y = y
        self.p = p
        self.type = type(p)

    def __call__(self,y=None,p=None):
        self.y = y
        self.p = p
        self.type = type(p)


    def loss(self):
        data = 0.5 * np.power((y.data - p.data), 2)
        requires_grad = self.p.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.p, self.gradient_SquareLoss)]
        else:
            depends_on = []
        return  self.type(data=data,requires_grad=requires_grad,depends_on=depends_on)


    def gradient_SquareLoss(self,grad):
        return -(self.y - self.p)

class CrossEntropy(Loss):
    def __init__(self,y=None,p=None):
        self.y = y
        self.p = p
        self.type = type(p)

    def __call__(self,y=None,p=None):
        self.y = y
        self.p = p
        self.type = type(p)

    def loss(self):
        # Avoid division by zero
        p = np.clip(self.p.data, 1e-15, 1 - 1e-15)
        data = - self.y.data * np.log(p) - (1 - self.y.data) * np.log(1 - p)
        requires_grad = self.p.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.p, self.gradient_CrossEntropy)]
        else:
            depends_on = []
        return  self.type(data=data,requires_grad=requires_grad,depends_on=depends_on)

    def gradient_CrossEntropy(self,grad):
        # Avoid division by zero
        p = np.clip(self.p.data, 1e-15, 1 - 1e-15)
        return - (self.y.data / p) + (1 - self.y.data) / (1 - p) * grad

    def acc(self, y, p):
        return accuracy_score(np.argmax(y.data, axis=1), np.argmax(p.data, axis=1))
