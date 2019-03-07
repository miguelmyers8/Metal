from metal.tensor import Tensor
from metal.parameter import Parameter
from metal.module import Module
from typing import Sequence, Iterator, Tuple, Any
import numpy as np

class Sequential(object):
    """ docstring for Sequential. takes a list of layers """

    def __init__(self, layers: Sequence[Any])-> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:

        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, x_test, y_test):

        pred = self.forward(x_test)
        for idx, item in enumerate(self.layers):
            if item.__class__.__name__ == "Dropout":
                self.layers.remove(item)

        m = x_test.shape[1]
        p = np.zeros((1, m))
        y = y_test.data

        for i in range(0, pred.shape[1]):
            if pred.data[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        print("Accuracy: "  + str(np.sum((p == y)/m)))
