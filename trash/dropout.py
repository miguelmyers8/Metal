"""File for Dropout (Inverted Dropout)
- This file computes forward pass of Dropout class:
- Dropout helps with overfitting.
- Dropout out layer takes in a int.
- which is the probability that a given hidden will be kept.
- so if int is 0.8 there is a 0.2 chance of eliminating any hidden unit.
"""

import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module
from autograd.tensor import Dependency


class Dropout(Module):

    # docstring for Dropout.
    def __init__(self, keepprob: int):
        super(Dropout, self).__init__()
        self.kp = keepprob

    def forward(self, inputs: Tensor):

        np.random.seed(inputs.shape[1]+1)

        self.inputs = inputs

        dropout_data =  np.random.rand(inputs.shape[0], inputs.shape[1])

        # the probability that a given hidden unit is kept.
        # for ever elemet of dropout_data that is equal to false that elemet is equal to (0)
        # for ever elemet of dropout_data that is equal to True that elemet is equal to (1)

        self.dropout_data = dropout_data < self.kp
        outputs = self.inputs.data * self.dropout_data

        # scaling up by dividing by keepprob (Inverted Dropout)
        outputs /= self.kp

        requires_grad = inputs.requires_grad

        # gradient function
        if requires_grad:
            def grad_dropout(grad: np.ndarray) -> np.ndarray:
                grad_d = grad * self.dropout_data
                grad_d = grad_d / self.kp
                return grad_d

            depends_on = [Dependency(inputs, grad_dropout)]
        else:
            depends_on = []

        #unfinished
        return Tensor(outputs, requires_grad, depends_on)
