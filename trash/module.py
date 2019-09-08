from typing import Iterator
import inspect

from autograd.tensor import Tensor
from autograd.parameter import Parameter


class Module:
    param = dict()
    grad = dict()

    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def rest_param(self):
        Module.param.clear()
        
    def rest_grad(self):
        Module.grad.clear()
