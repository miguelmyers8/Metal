
import time
from copy import deepcopy
from numpy.testing import assert_almost_equal
import unittest
from metal.initializers.activation_init import ActivationInitializer
from metal.utils.functions import random_stochastic_matrix, random_tensor
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_models import torch_gradient_generator

class activationstest(unittest.TestCase):
    def test_activations(self, N=50):
        print("Testing ReLU activation")
        time.sleep(1)
        test_relu_activation(N)
        test_relu_grad(N)


def test_relu_activation(N=None):
    from metal.initializers.activation_init import ActivationInitializer

    N = np.inf if N is None else N

    mine = ActivationInitializer('relu')()
    gold = lambda z: F.relu(torch.FloatTensor(z)).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1

def test_relu_grad(N=None):
    from metal.initializers.activation_init import ActivationInitializer

    N = np.inf if N is None else N

    mine = ActivationInitializer('relu')()
    gold = torch_gradient_generator(F.relu)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1
