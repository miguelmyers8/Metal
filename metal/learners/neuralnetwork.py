from __future__ import print_function, division
import numpy as _np
import matplotlib.pyplot as plt
from metal.utils.functions import batch_iterator
from metal.utils.misc import bar_widgets
import progressbar
from metal.learners.solver import Solver
from metal.autograd import numpy as np
import time


class NeuralNetwork(object):
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
    def __init__(self, optimizer, loss, validation_data=None, layers = []):
        self.optimizer = optimizer
        self.layers = layers
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

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

    def test_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X, retain_derived=False)
        loss = self.loss_function.loss(y, y_pred)
        acc = self.loss_function.acc(y, y_pred)
        return loss, acc


    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X, retain_derived=True)
        loss = self.loss_function.loss(y, y_pred)
        acc = self.loss_function.acc(y._value, y_pred._value)
        #gradient = self.loss_function.grad(y, y_pred)
        #Calculate the gradient of the loss function wrt y_pred
        self._backward_pass(loss=loss)
        #Update weights
        self._update_pass(loss=loss._value)
        return loss._value, acc

    def _forward_pass(self, X, retain_derived=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output, retain_derived)

        return layer_output

    def _backward_pass(self, loss):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        loss.backward()
        for layer in reversed(self.layers):
            layer.backward()

    def _update_pass(self, loss):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            layer.update(loss)

    def fit(self, X, y, n_epochs, batch_size):
        """ Trains the model for a fixed number of epochs """
        for epoch in range(n_epochs):
            train_batch_error, train_batch_acc, val_batch_error,val_batch_acc = [],[],[],[]
            start = time.time()
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, acc = self.train_on_batch(X_batch, y_batch)
                train_batch_error.append((loss * len(y_batch._value)))
                train_batch_acc.append((acc * len(y_batch._value)))

            self.errors["training"].append(_np.sum(train_batch_error)/len(X._value))

            if self.val_set is not None:
                val_loss, val_acc = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                val_batch_error.append((val_loss * len(self.val_set["y"])))
                val_batch_acc.append(( val_acc * len(self.val_set["y"])))

            self.errors["validation"].append(_np.sum(val_batch_error)/len(self.val_set["X"]))
            elapsed_time = time.time() - start

            print('Epoch: {}\ntrain_loss: {:.3f}, train_acc: {:.3f}%, | val_loss: {:.3f}, val_acc: {:.3f}%, time: {:.4f}[sec]'.format(
                epoch + 1, _np.sum(train_batch_error)/len(X._value),
                _np.sum(train_batch_acc)/len(X._value),
                _np.sum(val_batch_error)/len(self.val_set["X"]),
                _np.sum(val_batch_acc)/len(self.val_set["X"]),
                elapsed_time))
        return self.errors["training"], self.errors["validation"]


    def eval(self, X_test, y_test, plot_fit=False):
        if plot_fit:
            train_err, val_err = self.errors['training'], self.errors["validation"]
            # Training and validation error plot
            n = len(train_err)
            training, = plt.plot(range(n), train_err, label="Training Error")
            validation, = plt.plot(range(n), val_err, label="Validation Error")
            plt.legend(handles=[training, validation])
            plt.title("Error Plot")
            plt.ylabel('Error')
            plt.xlabel('Iterations')
            plt.show()
        _, accuracy = self.test_on_batch(X_test, y_test)
        print ("Accuracy:", accuracy)
        return accuracy

    def predict(self, X):
        pred = self._forward_pass(X, retain_derived=False)
        list_pred = pred.flatten().tolist()
        return list_pred.index(max(list_pred))

    def with_solver(self,data,**kwargs):
        self.solver = Solver(self,data,**kwargs)
