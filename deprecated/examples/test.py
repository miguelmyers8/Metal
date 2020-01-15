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

from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from memory_profiler import profile


optimizer = Adam()
data = datasets.load_digits()
X = data.data
y = data.target
loss = CrossEntropy
# Covnet to  one-hot encoding
y = to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, seed = 1)
print(X_train.shape)
X_train = X_train.reshape((-1,1,8,8))
X_test = X_test.reshape((-1,1,8,8))

X_train = Parameter(X_train, requires_grad=False)
X_test = Parameter(X_test, requires_grad=False)
y_train = Parameter(y_train, requires_grad=False)
y_test = Parameter(y_test, requires_grad=False)
print(X_train.shape)

X_train = Parameter(np.random.randn(1,3,256,256))
covnet = ConvNet(optimizer=optimizer, loss=loss)


covnet.add(Conv2D(n_filters=64, filter_shape=(11,11), stride=4, input_shape=(3,256,256), padding='valid', seed=1))
covnet.add(Activation('relu'))
covnet.add(BatchNormalization())
covnet.add(Conv2D(n_filters=50, filter_shape=(3,3), stride=1, padding='same', seed=5))
covnet.add(Activation('relu'))
covnet.add(BatchNormalization())
covnet.add(Conv2D(n_filters=50, filter_shape=(3,3), stride=1, padding='same', seed=2))
covnet.add(Flatten())
covnet.add(BatchNormalization())
covnet.add(Dense(256, seed=4))
covnet.add(Activation('relu'))
covnet.add(Dense(10,seed=3))
covnet.add(Activation('softmax'))

#covnet.eval(X_test, y_test)
#covnet.eval(X_test, y_test)
# python3 -m memory_profiler test.py with @profile
#
# mprof run --include-children python3 test.py && mprof plot --output memory-profile.png

def main():
    train_err, val_err = covnet.fit(X_train, y_train, n_epochs=10, batch_size=64)
    #covnet.eval(X_test, y_test)



    print("end")
if __name__ == "__main__":
    main()
