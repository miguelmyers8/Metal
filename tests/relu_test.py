from metal.layers.activation_functions import ReLU as rl
import unittest
from autograd.tensor import Tensor
from autograd.parameter import Parameter
import numpy as np
import torch

class TestRelu(unittest.TestCase):
    def test_Relu(self):
        np.random.seed(1)
        x = np.random.randn(5,4)
        t1 = Parameter(x)
        rl_= rl()
        stout=rl_(t1)
        stout.sum().backward()

        p1 = torch.tensor(t1.data,requires_grad=True)
        ps = torch.nn.ReLU()
        psout = ps(p1)

        psout.sum().backward()
        assert np.allclose(t1.grad.data,p1.grad.numpy())
