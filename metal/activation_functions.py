import numpy as np
from autograd.tensor import Tensor
from autograd.dependency import Dependency

# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + -x.exp())

class TanH():
    def __call__(self, x):
        return 2 / (1 + (-2 * x.exp())) - 1

class Softmax():
    def __call__(self, x):
        e_x = -x.max().exp()
        return e_x/ e_x.sum()

class ReLU():
    def __call__(self, tensor: Tensor) -> Tensor:
        self.inputs = tensor
        data = np.maximum(0, tensor.data)
        requires_grad = tensor.requires_grad
        if requires_grad:
            depends_on = [Dependency(tensor, self.grad_Relu)]
        else:
            depends_on = []
        return Tensor(data, requires_grad, depends_on)

        def grad_Relu(self, grad: np.ndarray) -> np.ndarray:
            grad[np.maximum(0, self.inputs.data) <= 0] = 0
            return grad

"""
class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

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
