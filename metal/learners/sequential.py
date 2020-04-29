from __future__ import print_function, division
import numpy as _np
import matplotlib.pyplot as plt
from metal.utils.utils import batch_iterator
import progressbar
from metal.learners.solver import Solver
from metal.autograd import numpy as np
import time
from metal.layers.module import Module

class Sequential(Module):
    pass
