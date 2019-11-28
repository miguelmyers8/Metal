import numpy as np
from autograd.tensor import Tensor
from autograd.dependency import Dependency

# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + (-x).exp())

class TanH():
    def __call__(self, x):
        return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

class ReLU():
    def __call__(self,x):
        return x * (x.data > 0)

class LeakyReLU():
    def __call__(self, x, alpha=0.2):
        y1 = (x * (x.data > 0))
        y2 = (x * alpha * (x.data <= 0))
        return y1 + y2

# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
class Softmax():
    """docstring for Softmax."""
    def __call__(self, x):
        shiftx = x - x.max()
        exps = shiftx.exp()
        return exps / exps.sum()
