
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
from torch_models import torch_gradient_generator, torch_xe_grad, TorchLinearActivation, TorchFCLayer, TorchConv2DLayer


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


class activationstest(unittest.TestCase):
    def test_activations(self, N=50):
        print("Testing ReLU activation")
        time.sleep(1)
        test_relu_activation(N)
        test_relu_grad(N)

        print("Testing Sigmoid activation")
        time.sleep(1)
        test_sigmoid_activation(N)
        test_sigmoid_grad(N)

        print("Testing Tanh activation")
        time.sleep(1)
        test_tanh_grad(N)

        print("Testing Softmax activation")
        time.sleep(1)
        test_softmax_activation(N)
        test_softmax_grad(N)

        print("Testing CrossEntropy loss")
        time.sleep(1)
        test_cross_entropy(N)
        test_cross_entropy_grad(N)

        print("Testing FullyConnected layer")
        time.sleep(1)
        test_FullyConnected(N)

        print("Testing Conv2D layer")
        time.sleep(1)
        test_Conv2D(N)



def test_tanh_grad(N=None):
    from metal.initializers.activation_init import ActivationInitializer

    N = np.inf if N is None else N

    mine = ActivationInitializer('tanh')()
    gold = torch_gradient_generator(F.tanh)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1






def test_softmax_activation(N=None):
    from metal.layers.softmax import Softmax

    N = np.inf if N is None else N

    mine = Softmax()
    gold = lambda z: F.softmax(torch.FloatTensor(z), dim=1).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.forward(z), gold(z))
        print("PASSED")
        i += 1



def test_softmax_grad(N=None):
    from metal.layers.softmax import Softmax
    from functools import partial

    np.random.seed(12345)

    N = np.inf if N is None else N
    p_soft = partial(F.softmax, dim=1)
    gold = torch_gradient_generator(p_soft)

    i = 0
    while i < N:
        mine = Softmax()
        n_ex = np.random.randint(1, 3)
        n_dims = np.random.randint(1, 50)
        z = random_tensor((n_ex, n_dims), standardize=True)
        out = mine.forward(z)

        assert_almost_equal(
            gold(z),
            mine.backward(np.ones_like(out)),
            err_msg="Theirs:\n{}\n\nMine:\n{}\n".format(
                gold(z), mine.backward(np.ones_like(out))
            ),
            decimal=3,
        )
        print("PASSED")
        i += 1








def test_sigmoid_activation(N=None):
    from metal.initializers.activation_init import ActivationInitializer

    N = np.inf if N is None else N

    mine = ActivationInitializer('sigmoid')()
    gold = expit

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1

def test_sigmoid_grad(N=None):
    from metal.initializers.activation_init import ActivationInitializer

    N = np.inf if N is None else N

    mine = ActivationInitializer('sigmoid')()
    gold = torch_gradient_generator(F.sigmoid)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1




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

def test_cross_entropy(N=None):
    from metal.losses.loss import CrossEntropy

    N = np.inf if N is None else N

    mine = CrossEntropy()
    gold = log_loss

    # ensure we get 0 when the two arrays are equal
    n_classes = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_one_hot_matrix(n_examples, n_classes)
    assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred))
    print("PASSED")

    # test on random inputs
    i = 1
    while i < N:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_one_hot_matrix(n_examples, n_classes)
        y_pred = random_stochastic_matrix(n_examples, n_classes)

        assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred, normalize=False))
        print("PASSED")
        i += 1


def test_cross_entropy_grad(N=None):
    from metal.losses.loss import CrossEntropy
    from metal.layers.softmax import Softmax

    N = np.inf if N is None else N

    mine = CrossEntropy()
    gold = torch_xe_grad
    sm = Softmax()

    i = 1
    while i < N:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)

        y = random_one_hot_matrix(n_examples, n_classes)

        # the cross_entropy_gradient returns the gradient wrt. z (NOT softmax(z))
        z = random_tensor((n_examples, n_classes))
        y_pred = sm.forward(z)

        assert_almost_equal(mine.grad(y, y_pred), gold(y, z), decimal=5)
        print("PASSED")
        i += 1




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




def test_Conv2D(N=None):
    from metal.layers.conv2D import Conv2D
    from metal.activations.activation import Tanh, ReLU, Sigmoid, Affine

    N = np.inf if N is None else N

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        f_shape = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        p, s = np.random.randint(0, 5), np.random.randint(1, 3)
        d = np.random.randint(0, 5)

        fr, fc = f_shape[0] * (d + 1) - d, f_shape[1] * (d + 1) - d
        out_rows = int(1 + (in_rows + 2 * p - fr) / s)
        out_cols = int(1 + (in_cols + 2 * p - fc) / s)

        if out_rows <= 0 or out_cols <= 0:
            continue

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Conv2D layer
        L1 = Conv2D(
            out_ch=n_out,
            kernel_shape=f_shape,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchConv2DLayer(
            n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
        )
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}".format(i))
        print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_out))
        print("dilation={}".format(d))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1
