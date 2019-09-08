import unittest
import pytest
from autograd.tensor import Tensor as tn
from autograd.linear import Linear
import torch
import numpy as np

class Testlinear(unittest.TestCase):
    def test_simple_linear_forward(self):
        np.random.seed(1)
        data = np.random.randn(2,2)
        x = tn(data, requires_grad=True)
        l = Linear(2,2)
        h = l.forward(x)
        h = h.data.round(decimals=4)
        h = np.float32(h)

        # pytorch
        d = torch.Tensor([[ 1.62434536, -0.61175641],[-0.52817175, -1.07296862]])
        w = torch.Tensor(l.w.data)
        b = torch.Tensor(l.b.data)
        o = w @ d + b
        o = o.numpy()
        o = o.round(decimals=4)
        assert h.tolist() == o.tolist()

    def test_simple_linear_backward(self):
        np.random.seed(1)
        data = np.random.randn(2,2)
        x = tn(data, requires_grad=True)
        l = Linear(2,2)
        h = l.forward(x)
        hs = h.sum()
        hs.backward()
        dl = l.w.grad / (1/data.shape[0])
        dl = dl.data.round(decimals=4)
        dl = np.float32(dl)

        # pytorch
        d = torch.Tensor([[ 1.62434536, -0.61175641],[-0.52817175, -1.07296862]])
        d.requires_grad_(True)
        w = torch.Tensor(l.w.data)
        w.requires_grad_(True)
        pb = torch.Tensor(l.b.data)
        pb.requires_grad_(True)
        o = w @ d + pb
        k = o.sum()
        k.backward()
        dw = w.grad
        dw = dw.numpy()
        dw = dw.round(decimals=4)

        assert dw.tolist() == dl.tolist()
