import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorPad(unittest.TestCase):
    def test_simple_pad(self):
        t1 = Tensor(np.random.randn(5,7), requires_grad=True, name="t1")
        t2 = t1.pad([(1,1),(1,1)],"constant")
        t3 = Tensor(np.random.randn(t2.shape[0],t2.shape[1]), requires_grad=True)
        t4 = t2 * t3
        t5 = t4.sum()
        t5.backward()
