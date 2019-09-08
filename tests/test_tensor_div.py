import unittest
import pytest
from autograd.tensor import Tensor

class TestTensorDiv(unittest.TestCase):
    def test_simple_Div(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 / t2

        assert t3.data.tolist() == [0.25, 0.4, 0.5]

        t4 = t3.sum()
        t4.backward()
        
        assert t1.grad.data.tolist() == [0.25, 0.2, 0.16666666666666666]
        assert t2.grad.data.tolist() == [-0.0625, -0.08, -0.08333333333333333]