from metal.layers.conv2D import Conv2D
import numpy as np
from metal.initializers.optimizer_init import OptimizerInitializer, Adam
import unittest
tcout = np.array([[[[0.44224389],
         [0.        ]],

        [[1.05154188],
         [0.        ]]],


       [[[0.        ],
         [0.3496756 ]],

        [[0.        ],
         [0.        ]]]])

class conv2Dtest(unittest.TestCase):
    def testconv2d(self):
        np.random.seed(1)
        c1 = Conv2D(1,(5,5),'same',stride=1,optimizer=Adam(),act_fn='relu')
        z=c1.forward(np.random.randn(2,2,2,3))
        assert np.allclose(z,tcout)
