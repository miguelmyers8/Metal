from tensor import Tensor as tn
import unittest

x = tn(5)

y = x + x

ls = [1,2,3,4]

xx = Tensor([1,2,3,4,5])

yy = Tensor([2,2,2,2,2])

class MyTest(unittest.TestCase):

    def test(self):
        self.assertEqual(y.data, 10)

    def test1(self):
        self.assertEqual(ls, [1,2,3,4])

    
