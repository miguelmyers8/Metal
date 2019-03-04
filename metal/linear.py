
import numpy as np
from metal.tensor import Tensor
from metal.parameter import Parameter
from metal.module import Module
from metal.tensor import Dependency


class Linear(Module):
    # docstring for Linear.

    def __init__(self, input_size: int, output_size: int) -> None:
        super(Linear, self).__init__()
        b = np.zeros((input_size,1))
        self.w = Parameter(input_size, output_size) * np.sqrt(2/output_size)
        self.b = Parameter(input = b)

    def forward(self, inputs: Tensor) -> Tensor:

        self.inputs = inputs
        m = self.inputs.shape[1]
        output = self.w.data @ self.inputs.data + self.b.data
        requires_grad = self.inputs.requires_grad or self.w.requires_grad or self.b.requires_grad
        depends_on: List[Dependency] = []

        # weight derivate
        if self.w.requires_grad:
            def grad_w(grad: np.ndarray) -> np.ndarray:
                    grad = (1./m) * (grad @ self.inputs.data.T)
                    return grad

            depends_on.append(Dependency(self.w, grad_w))

        # biase derivate
        if self.b.requires_grad:
            def grad_b(grad: np.ndarray) -> np.ndarray:
                grad = (1./m) * np.sum(grad, axis=1, keepdims=True)
                return grad

            depends_on.append(Dependency(self.b, grad_b))


        return Tensor(output,requires_grad,depends_on)
