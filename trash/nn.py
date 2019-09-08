from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module
from typing import Sequence, Iterator, Tuple, Any
import numpy as np


class Sequential(Module):
    """ docstring for Sequential. takes a list of layers """

    def __init__(self, layers: Sequence[Any]) -> None:
        self.layers = layers

    def add(self, layer): #adds a layer
        self.layers.append(layer)

    def remove(self, layer): # removes layer
        for idx, item in enumerate(self.layers):
            if item.__class__.__name__ == layer:
                self.layers.remove(item)


    def forward(self, inputs: Tensor) -> Tensor: #forward prop
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, x_test, y_test, layer=None): # predict

        if layer is not None:
            self.remove(layer)

        pred = self.forward(x_test)
        m = x_test.shape[1]
        p = np.zeros((1, m))
        y = y_test.data

        for i in range(0, pred.shape[1]):
            if pred.data[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.mean((p[0,:] == y[0,:]))))
