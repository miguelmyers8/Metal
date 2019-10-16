from metal.layers.activation_functions import Sigmoid as st
import unittest
import pytest
from autograd.tensor import Tensor
from autograd.parameter import Parameter
import numpy as np
import torch

class TestSigmoid(unittest.TestCase):
    def test_Sigmoid(self):
        np.random.seed(1)
        x = np.random.randn(5,4)
        t1 = Parameter(x)
        st_= st()
        stout=st_(t1)
        stout.sum().backward()

        p1 = torch.tensor(t1.data,requires_grad=True)
        ps = torch.nn.Sigmoid()
        psout = ps(p1)

        psout.sum().backward()
        assert np.allclose(t1.grad.data,p1.grad.numpy())
