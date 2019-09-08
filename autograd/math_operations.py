import numpy as np
from autograd.node import Node
from autograd.node import Dependency
import autograd.tensor


class Sum():

    def __init__(self, t: Node):
        self.t = t

    def _sum(self) -> Node:
        #Takes a Node and returns the 0-Node that's the sum of all its elements.
        data = self.t.data.sum()
        requires_grad = self.t.requires_grad

        if requires_grad:
            depends_on = [Dependency(self.t, self.grad_fn)]
        else:
            depends_on = []
        return autograd.tensor.Tensor(data, requires_grad, depends_on)

    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        #grad is necessarily a 0-Node, so each input element contributes that much
        return grad * np.ones_like(self.t.data)


class Add():

    def __init__(self, t1: Node, t2: Node):
        self.t1 = t1
        self.t2 = t2

    def _add(self) -> Node:
        data = self.t1.data + self.t2.data
        requires_grad = self.t1.requires_grad or self.t2.requires_grad
        depends_on: List[Dependency] = []

        if self.t1.requires_grad:
            depends_on.append(Dependency(self.t1, self.grad_fn1))
        if self.t2.requires_grad:
            depends_on.append(Dependency(self.t2, self.grad_fn2))
        return autograd.tensor.Tensor(data, requires_grad, depends_on)


    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        # Sum out added dims
        ndims_added = grad.ndim - self.t1.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        # Sum out added dims
        ndims_added = grad.ndim - self.t2.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

class Mul():
    def __init__(self, t1: Node, t2: Node):
        self.t1 = t1
        self.t2 = t2


    def _mul(self) -> Node:
        data = self.t1.data * self.t2.data
        requires_grad = self.t1.requires_grad or self.t2.requires_grad
        depends_on: List[Dependency] = []

        if self.t1.requires_grad:
            depends_on.append(Dependency(self.t1, self.grad_fn1))
        if self.t2.requires_grad:
            depends_on.append(Dependency(self.t2, self.grad_fn2))
        return autograd.tensor.Tensor(data, requires_grad, depends_on)


    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.t2.data
        # Sum out added dims
        ndims_added = grad.ndim - self.t1.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad


    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.t1.data
        # Sum out added dims
        ndims_added = grad.ndim - self.t2.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad


class Neg():
    def __init__(self, t: Node):
        self.t = t

    def _neg(self) -> Node:
        data = -self.t.data
        requires_grad = self.t.requires_grad
        if requires_grad:
            depends_on = [Dependency(self.t, lambda x: -x)]
        else:
            depends_on = []

        return autograd.tensor.Tensor(data, requires_grad, depends_on)

class Sub():
    def __init__(self, t1: Node, t2: Node):
        self.t1 = t1
        self.t2 = t2

    def _sub(self) -> Node:
        return self.t1 + -self.t2

class MatMul():
    def __init__(self, t1: Node, t2: Node):
        self.t1 = t1
        self.t2 = t2

    def _matmul(self) -> Node:
        """
        if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
        so grad3 is (n1, m2)
        if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
            grad1 = grad3 @ t2.T
            grad2 = t1.T @ grad3
        """
        data = self.t1.data @ self.t2.data
        requires_grad = self.t1.requires_grad or self.t2.requires_grad
        depends_on: List[Dependency] = []

        if self.t1.requires_grad:
            depends_on.append(Dependency(self.t1, self.grad_fn1))
        if self.t2.requires_grad:
            depends_on.append(Dependency(self.t2, self.grad_fn2))

        return autograd.tensor.Tensor(data, requires_grad, depends_on)

    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        return grad @ self.t2.data.T

    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        return self.t1.data.T @ grad

class Div():
    def __init__(self, t1: Node, t2: Node):
        self.t1 = t1
        self.t2 = t2

    def _div(self) -> Node:
        data = self.t1.data / self.t2.data
        requires_grad = self.t1.requires_grad or self.t2.requires_grad
        depends_on: List[Dependency] = []

        if self.t1.requires_grad:
            depends_on.append(Dependency(self.t1, self.grad_fn1))
        if self.t2.requires_grad:
            depends_on.append(Dependency(self.t2, self.grad_fn2))
        return autograd.tensor.Tensor(data, requires_grad, depends_on)

    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * (1 / self.t2.data)
        # Sum out added dims
        ndims_added = grad.ndim - self.t1.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * (-self.t1.data / (self.t2.data * self.t2.data))
        # Sum out added dims
        ndims_added = grad.ndim - self.t2.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(self.t2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
