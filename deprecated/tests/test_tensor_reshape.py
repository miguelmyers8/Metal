import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorReshape(unittest.TestCase):
    def test_simple_Reshape(self):
        t1 = Tensor([[1, 2, 3]], requires_grad=True, name="t1").reshape(3,1)
        t2 = Tensor([[4, 2, 6]], requires_grad=True, name="t2").reshape(3,1).T()
        t3 = t1@t2

        s = t3.sum()
        s.backward()

        assert t2.grad.data.tolist() == [[6., 6., 6.]]
        assert t1.grad.data.tolist() == [[12.],
                                         [12.],
                                         [12.]]
