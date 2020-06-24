from abc import ABC, abstractmethod
import numpy as _np
from ..autograd import numpy as np
from ..autograd import Container, primitive, defvjp

class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.forward(z)

    @abstractmethod
    def forward(self, z):
        pass

    @abstractmethod
    def backward(self, x, **kwargs):
        pass


class Softmax(ActivationBase):
    def __init__(self, dim=1):
        self.dim = dim

    @primitive
    def forward(self,X):
        """Actual computation of softmax forward pass"""
        # center data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(ans,self,X):
        def grad(gy):
            gx = ans * gy
            sumdx = gx.sum(axis=self.dim, keepdims=True)
            gx = gx - (ans * sumdx)
            return gx
        return grad
        
defvjp(Softmax.forward,Softmax.backward, argnums=(1,))


class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    @primitive
    def forward(self, z):
        return self.slope * z + self.intercept

    def backward(ans,self,x):
        def grad(gy):
            return self.slope * np.ones_like(x) * gy
        return grad
defvjp(Affine.forward,Affine.backward, argnums=(1,))


class ReLU(ActivationBase):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    @primitive
    def forward(self, z):
        return np.clip(z, 0, np.inf)

    def backward(ans,self,x):
        def grad(gy):
            return (x > 0).astype(dtype=gy.dtype) * gy
        return grad
defvjp(ReLU.forward,ReLU.backward, argnums=(1,))


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    @primitive
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(ans, self, x):
        def grad(gy):
            fn_x = ans
            return (fn_x * (1 - fn_x)) * gy
        return grad
defvjp(Sigmoid.forward,Sigmoid.backward, argnums=(1,))


class Tanh(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def forward(self, z):
        return np.tanh(z)

    def backward(ans, self, x):
        def grad(gy):
            return (1 - np.tanh(x) ** 2) * gy
        return grad
defvjp(Tanh.forward,Tanh.backward, argnums=(1,))
