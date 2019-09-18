import numpy as np
from autograd.dependency import Dependency

class Slice(object):
    """docstring for Slice."""
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


class T(object):
    """docstring for Transpose."""
    def __init__(self, t, axis):
        super(T, self).__init__()
        self.type = type(t)
        self.t = t
        self.axis = axis

    def _T(self):
        data = self.t.data.transpose(*self.axis)
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_t)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_t(self, grad: np.ndarray):
        return grad.transpose(*self.axis)

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


class Pad(object):
    """docstring for Pad."""
    def __init__(self, t, pad):
        super(Pad, self).__init__()
        self.type = type(t)
        self.t = t
        self.pad = pad

    def _pad(self):
        data = np.pad(self.t.data, self.pad[0], mode=self.pad[1])
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_pad)]
        else:
            depends_on = []
        return self.type(data, requires_grad, depends_on)

    def grad_pad(self, grad: np.array):
        slices = []
        for c in self.pad[0]:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return grad[tuple(slices)]

class Flatten(object):
    """docstring for Flatten."""
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
