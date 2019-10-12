import numpy as np
from autograd.tensor import Tensor
from autograd.dependency import Dependency

# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid(object):
    def __call__(self, x):
        return 1 / (1 + (-x).exp())

class TanH(object):
    def __call__(self, x):
        return (x.exp()-(-x).exp())/(x.exp()+(-x).exp())

class ReLU(object):
    def __call__(self,x):
        return x * (x.data > 0)

class LeakyReLU(object):
    def __call__(self, x, alpha=0.2):
        y1 = (x * (x.data > 0))
        y2 = (x * alpha *(x.data <= 0))
        return y1 + y2


"""
class Softmax():
    def __call__(self, x):
        x = x - x.max(axis=axis, keepdims=None)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=None)

class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    # Reference : https://arxiv.org/abs/1706.02515,
    # https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
"""
