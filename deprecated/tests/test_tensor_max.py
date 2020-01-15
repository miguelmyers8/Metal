import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorMax(unittest.TestCase):
    def test_simple_Max(self):
        t1 = Tensor([[1, 2, 3]], requires_grad=True)
        t2 = Tensor([[4, 5, 6]], requires_grad=True)
        j = t1.max() * 3
        g = j + t1
        s = g.sum()
        s.backward()

        assert t2.grad.data.tolist() == [[0., 0., 0.]]
        assert t1.grad.data.tolist() == [[ 1.,  1., 10.]]
