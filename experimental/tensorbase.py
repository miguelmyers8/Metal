from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union

import numpy as np


class Dependency(NamedTuple):
    TensorBase: "TensorBase"
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        arrayable = arrayable.astype(np.float32)
        return arrayable
    else:
        arr = np.array(arrayable)
        return arr.astype(np.float32)


TensorBaseable = Union["TensorBase", float, np.ndarray]


def ensure_TensorBase(TensorBaseable: TensorBaseable) -> "TensorBase":
    if isinstance(TensorBaseable, TensorBase):
        return TensorBaseable
    else:
        return TensorBase(TensorBaseable)


class TensorBase:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: List[Dependency] = None,
        id = None,
    ) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional["TensorBase"] = None

        if id is None:
            id = np.random.randint(0, 100_000)
        self.id = id

        if self.requires_grad:
            self.zero_grad()


    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None


    def zero_grad(self) -> None:
        self.grad = TensorBase(np.zeros_like(self.data, dtype=np.float32))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> "TensorBase":
        """gets called if I do t + other"""
        return _add(self, ensure_TensorBase(other))

    def __radd__(self, other) -> "TensorBase":
        """gets called if I do other + t"""
        return _add(ensure_TensorBase(other), self)

    def __iadd__(self, other) -> "TensorBase":
        """when we do t += other"""
        self.data = self.data + ensure_TensorBase(other).data
        return self

    def __isub__(self, other) -> "TensorBase":
        """when we do t -= other"""
        self.data = self.data - ensure_TensorBase(other).data
        return self

    def __imul__(self, other) -> "TensorBase":
        """when we do t *= other"""
        self.data = self.data * ensure_TensorBase(other).data
        return self

    def __mul__(self, other) -> "TensorBase":
        return _mul(self, ensure_TensorBase(other))

    def __rmul__(self, other) -> "TensorBase":
        return _mul(ensure_TensorBase(other), self)

    def __matmul__(self, other) -> "TensorBase":
        return _matmul(self, other)

    def __neg__(self) -> "TensorBase":
        return _neg(self)

    def __sub__(self, other) -> "TensorBase":
        return _sub(self, ensure_TensorBase(other))

    def __rsub__(self, other) -> "TensorBase":
        return _sub(ensure_TensorBase(other), self)

    def __getitem__(self, idxs) -> "TensorBase":
        return _slice(self, idxs)

    def __truediv__(self, other) -> "TensorBase":
        """gets called if I do t / other"""
        return _div(self, ensure_TensorBase(other))

    def __rtruediv__(self, other) -> "TensorBase":
        """gets called if I do other / t"""
        return _div(ensure_TensorBase(other), self)

    def backward(self, grad: "TensorBase" = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad TensorBase"

        if grad is None:
            if self.shape == ():
                grad = TensorBase(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-TensorBase")

        self.grad.data = self.grad.data + grad.data  # type: ignore

        for dependency in self.depends_on: #loop over the list
            backward_grad = dependency.grad_fn(grad.data) # apply gard fuction
            dependency.TensorBase.backward(TensorBase(backward_grad)) # get current tensorbase
                                                                      # apply backward function
                                                                      # wrapping the output gardent

    def sum(self) -> "TensorBase":
        return TensorBase_sum(self)


def TensorBase_sum(t: TensorBase) -> TensorBase:
    """
    Takes a TensorBase and returns the 0-TensorBase
    that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-TensorBase, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return TensorBase(data, requires_grad, depends_on)


def _add(t1: TensorBase, t2: TensorBase) -> TensorBase:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return TensorBase(data, requires_grad, depends_on)


def _mul(t1: TensorBase, t2: TensorBase) -> TensorBase:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return TensorBase(data, requires_grad, depends_on)


def _neg(t: TensorBase) -> TensorBase:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return TensorBase(data, requires_grad, depends_on)


def _sub(t1: TensorBase, t2: TensorBase) -> TensorBase:
    return t1 + -t2


def _matmul(t1: TensorBase, t2: TensorBase) -> TensorBase:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return TensorBase(data, requires_grad, depends_on)


def _slice(t: TensorBase, idxs) -> TensorBase:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            indices = (t.data[idxs] == t.data)
            bigger_grad = indices * grad
            return bigger_grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return TensorBase(data, requires_grad, depends_on)


def _div(t1: TensorBase, t2: TensorBase) -> TensorBase:

    data = t1.data / t2.data

    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * (1 / t2.data)

            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * (-t1.data / (t2.data * t2.data))

            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return TensorBase(data, requires_grad, depends_on)