
import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorTranspose(unittest.TestCase):
    def test_simple_Tranpose(self):
        t1 = Tensor([[1, 2, 3]], requires_grad=True, name="t1").T()
        t2 = Tensor([[4, 5, 6]], requires_grad=True, name="t2")
        t3 = t1@t2

        s = t3.sum()
        s.name = "s"

        s.backward()

        assert t2.grad.data.tolist() == [[6., 6., 6.]]
        assert t1.grad.data.tolist() == [[15.],[15.],[15.]]
