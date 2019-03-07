import numpy as np
from metal.tensor import Tensor


class Flatten(object):
    """docstring for Flatten."""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs : np.ndarray ) -> Tensor:

        if type(inputs) != Tensor:
            inputs = inputs.reshape(inputs.shape[0], -1 ).T
            print('data shape: '+str(inputs.shape))
            return Tensor(inputs)
        else:
            inputs = inputs.data
            self.forward(inputs)
