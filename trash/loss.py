from autograd.module import Module
from autograd.tensor import Tensor, Dependency
from autograd.linear import Linear
from autograd.functions import cel, L2_Regularization
import numpy as np


def CEL(predicted: Tensor, actual: Tensor, module: Module = None) -> Tensor:

    if module is None:
        return cel(predicted, actual, module)
    else:
        return cel(predicted, actual, module) + L2_Regularization(module)
