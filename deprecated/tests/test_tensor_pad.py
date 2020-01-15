import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorPad(unittest.TestCase):
    def test_simple_pad(self):
        np.random.seed(1)
        t1 = Tensor(np.array([[ 2.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.]]), requires_grad=True)
        t2 = t1.pad([(1,1),(1,1)],"constant")
        t3 = Tensor(np.array([[ 9.,  3.,  3.,  3.,  3.,3.,3.],
               [ 3.,  3.,  3.,  3.,  3.,3.,3.],
               [ 3.,  3.,  3.,  6.,  3.,3.,3.],
               [ 3.,  3.,  3.,  3.,  3.,3.,3.],
               [ 3.,  3.,  3.,  3.,  3.,3.,3.]]),requires_grad=True)
        t4=t2 * t3

        t5 = t4.sum()
        t5.backward()
        assert  np.allclose(t2.grad.data, np.array([[ 9.,  3.,  3.,  3.,  3.,3.,3.],
               [ 3.,  3.,  3.,  3.,  3., 3.,3.],
               [ 3.,  3.,  3.,  6.,  3., 3.,3.],
               [ 3.,  3.,  3.,  3.,  3., 3.,3.],
               [ 3.,  3.,  3.,  3.,  3., 3.,3.]],dtype=np.float32))

        assert  np.allclose(t1.grad.data, np.array([[3., 3., 3., 3., 3.],
       [3., 3., 6., 3., 3.],
       [3., 3., 3., 3., 3.]], dtype=np.float32))
