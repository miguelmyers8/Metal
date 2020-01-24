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

def torch_xe_grad(y, z):
    z = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
    y = torch.LongTensor(y.argmax(axis=1))
    loss = F.cross_entropy(z, y, size_average=False).sum()
    loss.backward()
    grad = z.grad.numpy()
    return grad


class TorchLinearActivation(nn.Module):
    def __init__(self):
        super(TorchLinearActivation, self).__init__()
        pass

    @staticmethod
    def forward(input):
        return input

    @staticmethod
    def backward(grad_output):
        return torch.ones_like(grad_output)


class TorchFCLayer(nn.Module):
    def __init__(self, n_in, n_hid, act_fn, params, **kwargs):
        super(TorchFCLayer, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hid)

        # explicitly set weights and bias
        # NB: we pass the *transpose* of the weights to pytorch, meaning
        # we'll need to check against the *transpose* of our outputs for
        # any function of the weights
        self.layer1.weight = nn.Parameter(torch.FloatTensor(params["W"].T))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(params["b"]))

        self.act_fn = act_fn
        self.model = nn.Sequential(self.layer1, self.act_fn)

    def forward(self, X):
        self.X = X
        if not isinstance(X, torch.Tensor):
            self.X = torchify(X)

        self.z1 = self.layer1(self.X)
        self.z1.retain_grad()

        self.out1 = self.act_fn(self.z1)
        self.out1.retain_grad()

    def extract_grads(self, X):
        self.forward(X)
        self.loss1 = self.out1.sum()
        self.loss1.backward()
        grads = {
            "X": self.X.detach().numpy(),
            "b": self.layer1.bias.detach().numpy(),
            "W": self.layer1.weight.detach().numpy(),
            "y": self.out1.detach().numpy(),
            "dLdy": self.out1.grad.numpy(),
            "dLdZ": self.z1.grad.numpy(),
            "dLdB": self.layer1.bias.grad.numpy(),
            "dLdW": self.layer1.weight.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        return grads



class TorchConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        super(TorchConv2DLayer, self).__init__()

        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        self.layer1 = nn.Conv2d(
            in_channels,
            out_channels,
            hparams["kernel_shape"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=hparams["dilation"] + 1,
            bias=True,
        )

        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -3, -4])
        assert self.layer1.weight.shape == W.shape
        assert self.layer1.bias.shape == b.flatten().shape

        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    def forward(self, X):
        # (N, H, W, C) -> (N, C, H, W)
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        # W (theirs): (n_out, n_in, f[0], f[1]) -> W (mine): (f[0], f[1], n_in, n_out)
        # X (theirs): (N, C, H, W)              -> X (mine): (N, H, W, C)
        # Y (theirs): (N, C, H, W)              -> Y (mine): (N, H, W, C)
        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads
