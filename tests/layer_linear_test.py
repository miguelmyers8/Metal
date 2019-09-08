from metal.layers import Dense
import unittest
import pytest
from autograd.tensor import Tensor
import numpy as np

class TestLayerDense(unittest.TestCase):
    def test_dense(self):
        np.random.seed(1)
        x = np.random.randn(3,2)

        d1 = Dense(n_units=3,input_shape=(2,))
        d1.initialize()
        f = d1.forward_pass(Tensor(x,True))

        testArray = np.array([[-0.51275393,  0.83436353, -0.82640448],
        [-0.51340049,  0.578796,  -0.03790986],
        [-1.32961756,  1.76497545, -0.9165615 ]]
        )
        assert np.allclose(f.data, testArray)

        s = f.sum()
        s.backward()

        testWGrad = np.array([[ 1.96158124 , 1.96158124 , 1.96158124],
        [-3.98626373, -3.98626373 ,-3.98626373]])
        assert np.allclose(d1.w.grad.data, testWGrad)
