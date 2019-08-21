from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np
from experimental.tensorbase import TensorBase

class Tensor(TensorBase):
    """docstring for Tensor."""

    def __init__(self, data: np.ndarray, requires_grad: bool = False, name: str = None):
        super().__init__(data,requires_grad)
        self.name = name
