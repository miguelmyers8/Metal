from __future__ import division
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np
from experimental.tensorbase import TensorBase
from experimental.tensorbase import Dependency

class Tensor(TensorBase):
    """docstring for Tensor."""

    def __init__(self, data: np.ndarray, requires_grad: bool = False, depends_on: List[Dependency] = None, name: str = None):
        super().__init__(data=data,requires_grad=requires_grad, depends_on=depends_on)
        self.name = name
