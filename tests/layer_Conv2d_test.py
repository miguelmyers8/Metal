from metal.conv2D import Conv2D
import unittest
import pytest
from autograd.tensor import Tensor
import numpy as np



class TestLayerConv2d(unittest.TestCase):
    def test_conv2d(self):
        np.random.seed(1)
        x = np.random.randn(3,2)
