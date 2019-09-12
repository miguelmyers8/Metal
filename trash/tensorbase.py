from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np
from autograd.node import ensure_Node, Node, Dependency
from autograd.math_operations import Sum, Add, MatMul, Mul, Sub, Div, Neg
from autograd.tensor_modifications import Slice, Transpose, Reshape, Flatten

class TensorBase(Node):
    """docstring for TensorBase."""

    def __init__(self, data: np.ndarray, requires_grad: bool = False, depends_on: List[Dependency] = None):
        super().__init__(data=data,requires_grad=requires_grad, depends_on=depends_on)

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

    def sum(self) -> "Node":
        return Sum(self)._sum()

    def T(self) -> "Node":
        return Transpose(self)._T()

    def reshape(self,*newshape)-> "Node":
        return Reshape(self,newshape)._reshape()

    def flatten(self)->"Node":
        return Flatten(self)._flatten()
