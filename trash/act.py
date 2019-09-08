import numpy as np

from autograd.tensor import Tensor, Dependency
from autograd.module import Module
import numpy as np

# Sigmoid
class Sigmoid(Module):

    # docstring for Sigmoid.
    def __init__(self,track_layer = False):
        super(Sigmoid, self).__init__()
        self.track_layer = False


    def forward(self, tensor: Tensor) -> Tensor:

        self.inputs = tensor
        data = 1 / (1 + np.exp(-tensor.data))
        requires_grad = tensor.requires_grad
        if requires_grad:

            def grad_Sigmoid(grad: np.ndarray) -> np.ndarray:
                return grad * (data * (1 - data))

            depends_on = [Dependency(tensor, grad_Sigmoid)]
        else:
            depends_on = []
        return Tensor(data, requires_grad, depends_on)


# Relu
class Relu(Module):

    # docstring for relu.
    def __init__(self, track_layer = False):
        super(Relu, self).__init__()
        self.track_layer = track_layer

    def forward(self, tensor: Tensor) -> Tensor:
        self.inputs = tensor
        data = np.maximum(0, tensor.data)
        requires_grad = tensor.requires_grad
        if requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                grad[data <= 0] = 0
                return grad

            depends_on = [Dependency(tensor, grad_fn)]
        else:
            depends_on = []
        return Tensor(data, requires_grad, depends_on)
