import numpy as np
from autograd.dependency import Dependency

class Slice(object):
    def __init__(self, t, idxs):
        self.type = type(t)
        self.t = t
        self.idxs = idxs

    def _slice(self):
        data = self.t.data[self.idxs]
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_slice)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_slice(self, grad: np.ndarray) -> np.ndarray:
        old_shape = self.t.shape
        new_grad = np.zeros(old_shape, dtype=np.float32)
        new_grad[self.idxs] = grad
        return new_grad


class Transpose(object):
    """docstring for Transpose."""
    def __init__(self, t):
        super(Transpose, self).__init__()
        self.type = type(t)
        self.t = t

    def _T(self):
        data = self.t.data.T
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_transpose)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_transpose(self, grad: np.ndarray):
        return grad.T

class Reshape(object):
    """docstring for Reshape."""

    def __init__(self, t, newshape):
        super(Reshape, self).__init__()
        self.type = type(t)
        self.new_shape = newshape
        self.t = t

    def _reshape(self):
        data = np.reshape(self.t.data, self.new_shape)
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_reshape)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_reshape(self, grad: np.array):
        old_shape = self.t.shape
        if len(self.new_shape) + 1 == len(grad.shape):
            old_shape = (grad.shape[0], *old_shape)
        return np.reshape(grad, old_shape)


class Flatten():
    def __init__(self, t):
        self.type = type(t)
        self.t = t

    def _flatten(self):
        data = np.reshape(self.t.data, (self.t.shape[0], -1))
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_flatten)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_flatten(self, grad: np.array):
        return np.reshape(grad, self.t.shape)
