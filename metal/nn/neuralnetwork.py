from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    #__slots__ = ('optimizer','layers','errors','loss_function','progressbar','val_set','trainable')
    """Neural Network. Deep Learning base model.
    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss
        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If the layer has weights that needs to be initialized
        layer(optimizer=self.optimizer)
        # Add layer to the network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X, retain_derived=True)
        loss = self.loss_function.loss(y, y_pred)
        acc = self.loss_function.acc(y, y_pred)
        #Calculate the gradient of the loss function wrt y_pred

        #Update weights

        return loss, acc

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output, training)

        return layer_output
