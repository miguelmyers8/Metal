import unittest
import pytest
from experimental.layers import Layer, Dense
from experimental.parameter import Parameter
from experimental.module import Module
from experimental.tensorbase import Dependency
import math
import numpy as np
from experimental.tensor import Tensor
from experimental.optimizers import StochasticGradientDescent as sgd
import torch
import torch.nn as nn
from torch.nn.functional import linear as pl


class dense_test(unittest.TestCase):
    def test_dense_forward(self):
        np.random.seed(1)
