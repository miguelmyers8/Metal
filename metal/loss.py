from metal.module import Module
from metal.tensor import Tensor, Dependency
from metal.linear import Linear
from metal.functions import cel, L2_Regularization
import numpy as np


def CEL(predicted: Tensor, actual: Tensor, module: Module = None) -> Tensor:

    if module is None:
        return cel(predicted, actual, module)
    else:
        return cel(predicted, actual, module) + L2_Regularization(module)
