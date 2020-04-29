import numpy as _np
from metal.utils.functions import accuracy_score,  is_binary, is_stochastic
from abc import ABC, abstractmethod
from metal.autograd import numpy as np


class ObjectiveBase(ABC):
    pass

class CrossEntropy(ObjectiveBase):
    pass
