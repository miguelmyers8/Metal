import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

#######################################################################
#       Gold-standard implementations for testing custom layers       #
#                       (Requires Pytorch)                            #
#######################################################################


def torchify(var, requires_grad=True):
    return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)


def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad
