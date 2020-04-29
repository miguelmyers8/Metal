from abc import ABC, abstractmethod
from ..autograd import numpy as np
from ..utils.function import softmax


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        pass

    @abstractmethod
    def grad(self, x, **kwargs):
        pass

class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        return self.slope * z + self.intercept

    def grad(self, x):
        return self.slope * np.ones_like(x)

    def grad2(self, x):
        return np.zeros_like(x)


class ReLU(ActivationBase):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

    def grad2(self, x):
        return np.zeros_like(x)

class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        fn_x = self.fn_x
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class Tanh(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Softmax(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Softmax"

    def fn(self, z, dim=-1):
        return  softmax(z, dim)

    def grad(self, x):
        pass

    def grad2(self, x):
        pass
