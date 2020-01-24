from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
import numpy as np

#from cs321


"""
A Solver encapsulates all the logic necessary for training classification
models. The Solver performs stochastic gradient descent using different
update rules defined in optim.py.
The solver accepts both training and validataion data and labels so it can
periodically check classification accuracy on both training and validation
data to watch out for overfitting.
To train a model, you will first construct a Solver instance, passing the
model, dataset, and various options (learning rate, batch size, etc) to the
constructor. You will then call the train() method to run the optimization
procedure and train the model.
After the train() method returns, model.params will contain the parameters
that performed best on the validation set over the course of training.
In addition, the instance variable solver.loss_history will contain a list
of all losses encountered during training and the instance variables
solver.train_acc_history and solver.val_acc_history will be lists of the
accuracies of the model on the training and validation set at each epoch.
Example usage might look something like this:
data = {
  'X_train': # training data
  'y_train': # training labels
  'X_val': # validation data
  'y_val': # validation labels
}
model = MyAwesomeModel(hidden_size=100, reg=10)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()
A Solver works on a model object that must conform to the following API:
- model.params must be a dictionary mapping string parameter names to numpy
  arrays containing parameter values.
- model.loss(X, y) must be a function that computes training-time loss and
  gradients, and test-time classification scores, with the following inputs
  and outputs:
  Inputs:
  - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
  - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
    label for X[i].
  Returns:
  If y is None, run a test-time forward pass and return:
  - scores: Array of shape (N, C) giving classification scores for X where
    scores[i, c] gives the score of class c for X[i].
  If y is not None, run a training time forward and backward pass and
  return a tuple of:
  - loss: Scalar giving the loss
  - grads: Dictionary with the same keys as self.params mapping parameter
    names to gradients of the loss with respect to those parameters.
"""

"""
Construct a new Solver instance.
Required arguments:
- model: A model object conforming to the API described above
- data: A dictionary of training and validation data containing:
  'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
  'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
  'y_train': Array, shape (N_train,) of labels for training images
  'y_val': Array, shape (N_val,) of labels for validation images
Optional arguments:
- update_rule: A string giving the name of an update rule in optim.py.
  Default is 'sgd'.
- optim_config: A dictionary containing hyperparameters that will be
  passed to the chosen update rule. Each update rule requires different
  hyperparameters (see optim.py) but all update rules require a
  'learning_rate' parameter so that should always be present.
- lr_decay: A scalar for learning rate decay; after each epoch the
  learning rate is multiplied by this value.
- batch_size: Size of minibatches used to compute loss and gradient
  during training.
- num_epochs: The number of epochs to run for during training.
- print_every: Integer; training losses will be printed every
  print_every iterations.
- verbose: Boolean; if set to false then no output will be printed
  during training.
- num_train_samples: Number of training samples used to check training
  accuracy; default is 1000; set to None to use entire training set.
- num_val_samples: Number of validation samples to use to check val
  accuracy; default is None, which uses the entire validation set.
- checkpoint_name: If not None, then save model checkpoints here every
  epoch.
"""


class Solver(object):

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']


        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)


        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()
