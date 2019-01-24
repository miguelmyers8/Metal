from tensor import Tensor
import unittest
import numpy as np


class MyTest(unittest.TestCase):

    def test(self):
        x = Tensor(5)
        y = x + x
        self.assertEqual(y.data, 10)

    def test1(self):
        ls = [1,2,3,4]
        self.assertEqual(ls, [1,2,3,4])

    def test2(self):
        a = Tensor([1,2,3,4,5], autograd=True)
        b = Tensor([2,2,2,2,2], autograd=True)
        c = Tensor([5,4,3,2,1], autograd=True)
        d = a + b
        e = b + c
        f = d + e
        f.backward(Tensor(np.array([1,1,1,1,1])))
        self.assertEqual(b.grad.data.tolist(), np.array([2,2,2,2,2]).tolist())
