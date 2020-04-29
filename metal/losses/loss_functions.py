import numpy as _np
from .utils.utils import accuracy_score

from metal.autograd import numpy as np
from metal.autograd import Container

class Loss(object):
    pass

class SquareLoss(Loss):
    pass

class CrossEntropy(Loss):
    pass
