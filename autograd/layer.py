import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module
from autograd.tensor import Dependency

class Layer(Module):

    #docstring for Layer.
    def __init__(self):
        super(Layer, self).__init__()
        self.inputs = None

    #forward pass
    def forward(self, inputs: Tensor)->Tensor:
        raise NotImplementedError

    def test(self):
        return "self.inputs"

class Linear(Layer):

    # docstring for Linear.
    def __init__(self, input_size: int, output_size: int) -> None:

        super(Linear, self).__init__()
        b = np.zeros((input_size, 1))
        self.w = Parameter(input_size, output_size)
        self.b = Parameter(inputs=b)

    def forward(self, inputs: Tensor) -> Tensor:

        self.inputs = inputs
        m = self.inputs.shape[1]
        output = self.w.data @ self.inputs.data + self.b.data
        requires_grad = (
            self.inputs.requires_grad or self.w.requires_grad or self.b.requires_grad
        )
        depends_on: List[Dependency] = []

        # weight derivate
        if self.w.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                grad = (1.0 / m) * (grad @ self.inputs.data.T)
                return grad

            depends_on.append(Dependency(self.w, grad_fn))

        # activation derivate
        if self.inputs.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                grad = self.w.data.T @ grad
                return grad

            depends_on.append(Dependency(self.inputs, grad_fn))

        # biase derivate
        if self.b.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                grad = (1.0 / m) * np.sum(grad, axis=1, keepdims=True)
                return grad

            depends_on.append(Dependency(self.b, grad_fn))

        return Tensor(output, requires_grad, depends_on)


class Sigmoid(Layer):

    #docstring for Sigmoid.
    def __init__(self):
        super(Sigmoid, self).__init__()

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



class Relu(Layer):

    #docstring for relu.
    def __init__(self):
        super(Relu, self).__init__()

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


class Flatten(Layer):

    #docstring for Flatten.
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs: np.ndarray) -> Tensor:

        if type(inputs) != Tensor:
            inputs = inputs.reshape(inputs.shape[0], -1).T
            print("data shape: " + str(inputs.shape))
            return Tensor(inputs)
        else:
            inputs = inputs.data
            self.forward(inputs)
