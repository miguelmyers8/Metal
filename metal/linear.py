import numpy as np
from metal.tensor import Tensor
from metal.parameter import Parameter
from metal.module import Module
from metal.tensor import Dependency


class Linear(Module):


    lambda_ = None

    # docstring for Linear.
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Linear, self).__init__()
        b = np.zeros((input_size, 1))
        self.w = Parameter(input_size, output_size)
        self.b = Parameter(inputs=b)


    # computes z = W @ X + b
    # find required graident
    # outputs z, required graident
    def _forward(self,inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.m = self.inputs.shape[1]
        output = self.w.data @ self.inputs.data + self.b.data
        requires_grad = (
            self.inputs.requires_grad or self.w.requires_grad or self.b.requires_grad
        )
        return output, requires_grad

    # weight derivate
    # input gradient
    # output gradient
    def grad_fn_w(self, grad: np.ndarray) -> np.ndarray:
        grad = (1.0 / self.m) * (grad @ self.inputs.data.T)
        return grad

    # l2 derivate
    # input gradient
    # output gradient
    def grad_fn_l2(self, grad: np.ndarray) -> np.ndarray:
        grad = (1.0 / self.m) * (grad @ self.inputs.data.T) + (self.lambda_ * self.w.data) / self.m
        return grad

    # activation derivate
    # input gradient
    # output gradient
    def grad_fn_a(self, grad: np.ndarray) -> np.ndarray:
        grad = self.w.data.T @ grad
        return grad

    # biase derivate
    # input gradient
    # output gradient
    def grad_fn_b(self, grad: np.ndarray) -> np.ndarray:
        grad = (1.0 / self.m) * np.sum(grad, axis=1, keepdims=True)
        return grad

    # main function
    def forward(self, inputs: Tensor) -> Tensor:

        output, requires_grad = self._forward(inputs)
        depends_on: List[Dependency] = []

        if self.w.requires_grad:

            if self.lambda_ is None:
                # applying normal weight gradient
                depends_on.append(Dependency(self.w, self.grad_fn_w))

            else:
                # applying lambda gradient
                depends_on.append(Dependency(self.w, self.grad_fn_l2))


        if self.inputs.requires_grad:
            # apply activation gradient
            depends_on.append(Dependency(self.inputs, self.grad_fn_a))


        if self.b.requires_grad:
            # apply biase gradient
            depends_on.append(Dependency(self.b, self.grad_fn_b))

        return Tensor(output, requires_grad, depends_on)
