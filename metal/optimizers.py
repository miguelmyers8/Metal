# Optimizers for models that use gradient based methods for finding the
# weights that minimizes the loss.
# A great resource for understanding these methods:
# http://sebastianruder.com/optimizing-gradient-descent/index.html

import numpy as np
from autograd.tensor import Tensor

class StochasticGradientDescent():
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None

    def update(self, w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w.data))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * w.grad
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt
