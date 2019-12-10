from __future__ import print_function

from metal.utils.data_loader.data_loader import Data_loader
from metal.nn import NeuralNetwork
from metal.utils import train_test_split, to_categorical, normalize
from metal.utils import get_random_subsets, shuffle_data, Plot
from metal.utils.data_operation import accuracy_score
from metal.optimizers import StochasticGradientDescent,Adam
from metal.loss_functions import CrossEntropy
from metal.utils.misc import bar_widgets
from metal.layers.dense import Dense
from metal.layers.conv2D import Conv2D
from metal.layers.flatten import Flatten
from metal.layers.layer import Activation
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from metal.models.convnet import ConvNet
import h5py
from metal.layers.batchnormalization_ import BatchNormalization
from metal.utils.production_util import save_model

import matplotlib.pyplot as plt
import math
import numpy as np
da = Data_loader()
x,y = da.create_training_data(classes=['dogs','cats'])
optimizer = Adam()
X = x
print(y.count(1)," ",y.count(0))
y = np.array(y)
loss = CrossEntropy
print(X.shape,"  ",y.shape)
# Covnet to  one-hot encoding
y = to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, seed = 1)

X_train = X_train.reshape((-1,1,50,50))/255.0
X_test = X_test.reshape((-1,1,50,50))/255.0

X_train = Parameter(X_train, requires_grad=False)
X_test = Parameter(X_test, requires_grad=False)
y_train = Parameter(y_train, requires_grad=False)
y_test = Parameter(y_test, requires_grad=False)

covnet = ConvNet(optimizer=optimizer, loss=loss,  validation_data=(X_test,y_test))

covnet.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1,50,50), padding='same', seed=1))
covnet.add(Activation('relu'))
covnet.add(BatchNormalization())
covnet.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, padding='same', seed=2))
covnet.add(Flatten())
covnet.add(BatchNormalization())
covnet.add(Dense(256, seed=4))
covnet.add(Activation('relu'))
covnet.add(Dense(2,seed=3))
covnet.add(Activation('softmax'))

train_err, val_err = covnet.fit(X_train, y_train, n_epochs=10, batch_size=64)
covnet.eval(X_test, y_test)
