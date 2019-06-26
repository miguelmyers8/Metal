from autograd.module import Module
from autograd.tensor import Tensor, Dependency
from autograd.linear import Linear
import numpy as np

# cross entropy loss
def cel(predicted: Tensor, actual: Tensor, module: Module = None) -> Tensor:
    global m
    m = actual.shape[1]
    data = (-1.0 / m) * sum(actual.data * np.log(predicted.data) + (1-actual.data) * np.log(1-predicted.data))
    requires_grad = predicted.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (
                (
                    np.divide(predicted.data-actual.data,m*(predicted.data-predicted.data*predicted.data))
                    #- np.divide(1 - actual.data, np.log(10) * (1 - predicted.data))
                )


            )
        depends_on = [Dependency(predicted, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


# l2 Regularization
def L2_Regularization(module: Module = None):
    global sum_
    sum_ = 0
    for layer in module.layers:
        if type(layer) == Linear:
            sum_ += Linear.lambda_ * (np.sum(np.square(layer.w.data))) / (2 * m)
    return sum_
