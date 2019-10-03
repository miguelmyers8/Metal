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
        arrayable =  arrayable.astype(np.float64, copy=False)
        return arrayable
    else:
        return np.array(arrayable).astype(np.float64, copy=False)


Nodeable = Union["Node", float, np.ndarray]


def ensure_Node(Nodeable: Nodeable) -> "Node":
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
        self.grad: Optional["Node"] = None
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
        self.grad = Node(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self) -> str:
        return f"Node({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> "Node":
        """gets called if I do t + other"""
        return Add(self,ensure_Node(other))._add()

    def __radd__(self, other) -> "Node":
        """gets called if I do other + t"""
        return Add(ensure_Node(other), self)._add()

    def __iadd__(self, other) -> "Node":
        """when we do t += other"""
        self.data = self.data + ensure_Node(other).data
        return self

    def __isub__(self, other) -> "Node":
        """when we do t -= other"""
        self.data = self.data - ensure_Node(other).data
        return self

    def __imul__(self, other) -> "Node":
        """when we do t *= other"""
        self.data = self.data * ensure_Node(other).data
        return self

    def __mul__(self, other) -> "Node":
        return Mul(self, ensure_Node(other))._mul()

    def __rmul__(self, other) -> "Node":
        return Mul(ensure_Node(other), self)._mul()

    def __matmul__(self, other) -> "Node":
        return MatMul(self, other)._matmul()

    def __neg__(self) -> "Node":
        return Neg(self)._neg()

    def __sub__(self, other) -> "Node":
        return Sub(self, ensure_Node(other))._sub()

    def __rsub__(self, other) -> "Node":
        return Sub(self, ensure_Node(other))._sub()

    def __getitem__(self, idxs) -> "Node":
        return Slice(self, idxs)._slice()

    def __truediv__(self, other) -> "Node":
        """gets called if I do t / other"""
        return Div(self, ensure_Node(other))._div()

    def __rtruediv__(self, other) -> "Node":
        """gets called if I do other / t"""
        return Div(ensure_Node(other), self)._div()

    def sum(self,axis=None) -> "Node":
        return Sum(self,axis)._sum()

    def T(self,*axis) -> "Node":
        return T(self,axis)._T()

    def pad(self,*pad) -> "Node":
        return Pad(self,pad)._pad()

    def reshape(self,*newshape)-> "Node":
        return Reshape(self,newshape)._reshape()

    def exp(self)->"Node":
        return Exp(self)._exp()

    def max(self)->"Node":
        return Max(self)._max()

    def backward(self, grad: "Node" = None) -> None:
        Autograd(self).backward(grad) # apply backward function wrapping the output gardent
