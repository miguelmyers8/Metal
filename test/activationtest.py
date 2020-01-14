import unittest
from metal.initializers.activation_init import ActivationInitializer
import numpy as np

actoutrelu_ = np.array([[1.62434536, 0.,         0.        ],
                        [0.,         0.86540763, 0.        ]])

actaffine_ = np.array([[ 1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862,  0.86540763, -2.3015387 ]])

class activationstest(unittest.TestCase):
    def testrelu(self):
        np.random.seed(1)
        act = ActivationInitializer('relu')()
        actoutrelu = act(np.random.randn(2,3))
        assert np.allclose(actoutrelu,actoutrelu_)

    def testaffine(self):
        np.random.seed(1)
        act = ActivationInitializer('Affine(slope=1, intercept=0)')()
        actaffine = act(np.random.randn(2,3))
        assert np.allclose(actaffine,actaffine_)
