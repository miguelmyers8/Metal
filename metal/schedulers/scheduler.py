from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from math import erf
from .utils.utils import gaussian_cdf


class SchedulerBase(ABC):
        pass


class ConstantScheduler(SchedulerBase):
    pass
