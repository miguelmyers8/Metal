
import time
from copy import deepcopy
from numpy.testing import assert_almost_equal
import unittest
from metal.initializers.activation_init import ActivationInitializer
from metal.utils.functions import random_stochastic_matrix, random_tensor, random_one_hot_matrix
from metal.optimizers.optimizer import Adam
import numpy as np
from scipy.special import expit

from sklearn.metrics import log_loss, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_models import torch_gradient_generator, torch_xe_grad, TorchLinearActivation, TorchFCLayer




def torchify(var, requires_grad=True):
    return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)

def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


def test_FullyConnected(N=None):
    from metal.layers.dense import Dense
    from metal.activations.activation import Tanh, ReLU, Sigmoid, Affine

    N = np.inf if N is None else N

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        X = random_tensor((n_ex, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize FC layer
        L1 = Dense(n_out=n_out, act_fn=act_fn)

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchFCLayer(n_in, n_out, torch_fn, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"].T, "W"),
            (L1.parameters["b"], "b"),
            (dLdy, "dLdy"),
            (L1.gradients["W"].T, "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}\nact_fn={}".format(i, act_fn_name))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


print("Testing FullyConnected layer")
test_FullyConnected(N=50)
