import numpy as np
from autograd.node import Node
from autograd.node import Dependency
import autograd.tensor


class Slice():
    def __init__(self, t: Node, idxs):
        self.t = t
        self.idxs = idxs

    def _slice(self) -> Node:
        data = self.t.data[self.idxs]
        requires_grad = self.t.requires_grad

        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_fn)]
        else:
            depends_on = []

        return autograd.tensor.Tensor(data, requires_grad, depends_on)

    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        old_shape = self.t.shape
        new_grad = np.zeros(old_shape, dtype=np.float64)
        new_grad[self.idxs] = grad
        return new_grad


class Transpose(object):
    """docstring for Transpose."""
    def __init__(self, t: Node):
        super(Transpose, self).__init__()
        self.t = t

    def _T(self):
        data = self.t.data.T
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_T)]
        else:
            depends_on = []
        return autograd.tensor.Tensor(data, requires_grad, depends_on)

    def grad_T(self, grad: np.ndarray):
        return grad.T
