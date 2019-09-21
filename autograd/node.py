from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np
from autograd.math_operations import Sum, Add, MatMul, Mul, Sub, Div, Neg, Exp, Max
from autograd.tensor_modifications import Slice, T, Reshape, Pad
from autograd.dependency import Dependency
from autograd.engin import  Autograd


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        arrayable =  arrayable.astype(np.float32, copy=False)
        return arrayable
    else:
        return np.array(arrayable).astype(np.float32, copy=False)


Nodeable = Union["Tensor", float, np.ndarray]


def ensure_Node(Nodeable: Nodeable) -> "Tensor":
    if isinstance(Nodeable, Node):
        return Nodeable
    else:
        return Node(Nodeable)


class Node(object):
    def __init__(self,data: Arrayable,requires_grad: bool = False,depends_on: List[Dependency] = None, id = None,) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional["Tensor"] = None
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
        self.grad = Node(np.zeros_like(self.data, dtype=np.float32))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> "Tensor":
        """gets called if I do t + other"""
        return Add(self,ensure_Node(other))._add()

    def __radd__(self, other) -> "Tensor":
        """gets called if I do other + t"""
        return Add(ensure_Node(other), self)._add()

    def __iadd__(self, other) -> "Tensor":
        """when we do t += other"""
        self.data = self.data + ensure_Node(other).data
        return self

    def __isub__(self, other) -> "Tensor":
        """when we do t -= other"""
        self.data = self.data - ensure_Node(other).data
        return self

    def __imul__(self, other) -> "Tensor":
        """when we do t *= other"""
        self.data = self.data * ensure_Node(other).data
        return self

    def __mul__(self, other) -> "Tensor":
        return Mul(self, ensure_Node(other))._mul()

    def __rmul__(self, other) -> "Tensor":
        return Mul(ensure_Node(other), self)._mul()

    def __matmul__(self, other) -> "Tensor":
        return MatMul(self, other)._matmul()

    def __neg__(self) -> "Tensor":
        return Neg(self)._neg()

    def __sub__(self, other) -> "Tensor":
        return Sub(self, ensure_Node(other))._sub()

    def __rsub__(self, other) -> "Tensor":
        return Sub(self, ensure_Node(other))._sub()

    def __getitem__(self, idxs) -> "Tensor":
        return Slice(self, idxs)._slice()

    def __truediv__(self, other) -> "Tensor":
        """gets called if I do t / other"""
        return Div(self, ensure_Node(other))._div()

    def __rtruediv__(self, other) -> "Tensor":
        """gets called if I do other / t"""
        return Div(ensure_Node(other), self)._div()

    def sum(self) -> "Tensor":
        return Sum(self)._sum()

    def T(self,*axis) -> "Tensor":
        return T(self,axis)._T()

    def pad(self,*pad) -> "Tensor":
        return Pad(self,pad)._pad()

    def reshape(self,*newshape)-> "Tensor":
        return Reshape(self,newshape)._reshape()

    def exp(self)->"Tensor":
        return Exp(self)._exp()

    def max(self)->"Tensor":
        return Max(self)._max()

    def backward(self, grad: "Tensor" = None) -> None:
        Autograd(self).backward(grad) # apply backward function wrapping the output gardent
