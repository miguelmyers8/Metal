import unittest
from autograd.tensor import Tensor
import numpy as np
class TestTensorExp(unittest.TestCase):
    def test_simple_exp(self):
        t1 = Tensor([1, 2, 3], requires_grad=True).exp()
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        assert  np.allclose(t3.data, np.array([[ 10.873127 , 36.945282 ,120.51322 ]]))

        t4 = t3.sum()
        t4.backward()

        assert np.allclose(t1.grad.data, np.array([[4., 5. ,6.]]))
        assert np.allclose(t2.grad.data, np.array([[ 2.7182817,  7.389056 , 20.085537 ]]))
