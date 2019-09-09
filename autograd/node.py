from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np


class Dependency(NamedTuple):
    Node: "Node"
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Nodeable = Union["Node", float, np.ndarray]


def ensure_Node(Nodeable: Nodeable) -> "Node":
    if isinstance(Nodeable, Node):
        return Nodeable
    else:
        return Node(Nodeable)


class Node:
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
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: "Node" = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad Node"

        if grad is None:
            if self.shape == ():
                grad = Node(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-Node")
                
        self.grad.data = self.grad.data + grad.data  # type: ignore

        for dependency in self.depends_on: #loop over the list
            backward_grad = dependency.grad_fn(grad.data) # apply gard fuction
            dependency.Node.backward(Node(backward_grad)) # get current Node
                                                                      # apply backward function
                                                                      # wrapping the output gardent
