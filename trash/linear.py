
"""This is Linear file.

- This file computes forward pass of Linear class:
- forward = W @ X + b

- @ is matrix multiplaction opperater.
- X is a (Nx,M) dim matrix
- W is (c,Nx) dim matrix

- gradients for Linear class:
- dl/dW, dl/L2, dl/dX, dl/db
- dl/dW = grad_fn_w
- dl/L2 = grad_fn_l2
- dl/dX = grad_fn_a
- dl/db = grad_fn_b
- The gradients are written in this class and are not coming from Tensor
"""

import numpy as np
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module
from autograd.tensor import Dependency


class Linear(Module):


    # l2 norm:
        # helps with over fitting/high variance
        # If lambda_ is really big. hidden units in W will be close to 0.
        # therefore it will produce a much simpler NN.
    lambda_ = None

    # docstring for Linear.
    # Linear takes in int for input_size and output_size and np.ndarray for data_in.
    def __init__(self, input_size: int = None,
                        output_size: int = None,
                        data_in_w: np.ndarray = None,
                        data_in_b: np.ndarray = None,
                        track_layer: bool = True
                        ) -> None:

        super(Linear, self).__init__()

        self.track_layer = track_layer  #track linear to get parameters and at optimzation time.

        if (input_size and output_size is not None):
            b = np.zeros((input_size, 1))
            self.w = Parameter(input_size,output_size)
            self.b = Parameter(inputs_ = b)

        elif((type(data_in_w) and type(data_in_b)) is np.ndarray):
            #b = np.zeros((data_in_w.shape[0], 1))
            self.w = Parameter(inputs_ = data_in_w)
            self.b = Parameter(inputs_ = data_in_b)



    #computes z = W @ X + b
    def forward(self, inputs: Tensor) -> Tensor:

        self.inputs = inputs
        m = self.inputs.shape[1]

        output = self.w.data @ self.inputs.data + self.b.data
        requires_grad = (self.inputs.requires_grad or self.w.requires_grad or self.b.requires_grad)
        depends_on: List[Dependency] = []

        if self.w.requires_grad:

            if self.lambda_ is None:
                # applying normal weight gradient
                def grad_fn_w(grad: np.ndarray) -> np.ndarray:
                    grad = (1.0 / m) * (grad @ self.inputs.data.T)
                    return grad

                depends_on.append(Dependency(self.w, grad_fn_w))
            else:
                # applying lambda gradient
                def grad_fn_l2(grad: np.ndarray) -> np.ndarray:
                    grad = (1.0 / m) * (grad @ self.inputs.data.T) + (
                        self.lambda_ * self.w.data
                    ) / m
                    return grad

                depends_on.append(Dependency(self.w, grad_fn_l2))

        if self.inputs.requires_grad:
            # apply activation gradient
            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                grad = self.w.data.T @ grad
                return grad

            depends_on.append(Dependency(self.inputs, grad_fn_a))

        if self.b.requires_grad:
            # apply biase gradient
            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                grad = (1.0 / m) * np.sum(grad, axis=1, keepdims=True)
                return grad

            depends_on.append(Dependency(self.b, grad_fn_b))

        return Tensor(output, requires_grad, depends_on)
