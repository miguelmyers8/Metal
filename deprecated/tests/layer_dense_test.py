from metal.layers.dense import Dense
import unittest
from autograd.tensor import Tensor
import numpy as np


testArray = np.array([[-0.01960187,  0.8117337,  -0.79563284],
 [ 0.36192212, 0.37144417,  0.99197656],
 [ 0.54182684,  1.4194115,   0.71508807]]
)

testWGrad = np.array([[ 1.9615812, 1.9615812,  1.9615812],
 [-3.9862638, -3.9862638, -3.9862638]])


class TestLayerDense(unittest.TestCase):
    def test_dense(self):
        np.random.seed(1)
        x = np.random.randn(3,2)

        d1 = Dense(n_units=3,input_shape=(2,),seed=1)
        d1.initialize()
        f = d1.forward_pass(Tensor(x,True))

        assert np.allclose(f.data, testArray)

        s = f.sum()
        s.backward()

        assert np.allclose(d1.w.grad.data, testWGrad)
