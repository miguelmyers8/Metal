import unittest
from autograd.tensor import Tensor
from autograd.parameter import Parameter
import numpy as np
import torch

from metal.layers.dense import Dense
from metal.layers.flatten import Flatten
from metal.nn import NeuralNetwork
from metal.layers.conv2D import Conv2D
from metal.optimizers import Adam
from metal.loss_functions import CrossEntropy


true_c3out = np.array([[[[-0.79899037,  1.0320021 ,  0.23549885],
         [ 0.2674513 ,  0.9715072 ,  0.53558135],
         [ 0.3240632 , -0.00254381,  0.33129525]],

        [[-0.21604921,  0.48240322, -0.00326236],
         [-0.6017107 , -1.1812458 ,  0.16737139],
         [-0.72112125, -0.40808514, -0.32228673]]],


       [[[-0.4251591 ,  0.683436  , -0.24868044],
         [-0.33330798,  0.0398453 ,  0.18328846],
         [ 0.49960458,  0.35438943, -0.51435834]],

        [[ 1.5021229 , -0.7569632 , -0.32508022],
         [ 1.0651203 , -0.80910695, -0.24125695],
         [ 0.4442642 ,  0.83503157,  0.17313306]]]], dtype=np.float32)

true_c4out = np.array([[[[-0.42370066, -0.43009187, -0.50845811],
         [ 0.25508575, -0.575298  , -0.10159585],
         [-0.06149733, -0.11085947,  0.06543428]],

        [[-0.22442041, -0.08442765, -0.51309691],
         [ 0.00406582, -0.6208603 , -0.24300702],
         [ 0.24924928,  0.03614051,  0.41039784]]],


       [[[ 0.34074468, -0.23730474, -0.03891834],
         [ 0.55066421, -0.45215798,  0.41334732],
         [-0.06129876, -0.34717051, -0.14316889]],

        [[-0.4970375 ,  0.30046468, -0.09504793],
         [-0.45896246, -0.37225861,  0.40149224],
         [-0.15349387, -0.37440159,  0.08442678]]]])

class Testnn(unittest.TestCase):
    def test_nn(self):
        optimizer = Adam()
        clf = NeuralNetwork(optimizer=optimizer,loss=CrossEntropy)
        clf.add(Conv2D(n_filters=2, filter_shape=(3,3), stride=1, input_shape=(3,3,3), padding='same',seed=1))
        clf.add(Conv2D(n_filters=2, filter_shape=(3,3), stride=1, padding='same',seed=2))
        clf.add(Flatten())
        clf.add(Dense(1,seed=1))

        np.random.seed(4)
        x = np.random.randn(2,3,3,3)
        y = np.array([[1],[0]]).reshape(-1,1)

        _x_ = Tensor(x,True)
        _y_ = Tensor(y,True)


        clf.fit(_x_,_y_,n_epochs=1,batch_size=2)

        assert np.allclose(clf.layers[0].output, true_c3out)
        assert np.allclose(clf.layers[1].output, true_c4out)

        trainFT = [l.trainable for l in clf.layers]
        trainftv = any(a == True for a in trainFT)
        assert trainftv == True

        clf.set_trainable(False)

        trainFT = [l.trainable for l in clf.layers]
        trainftv = any(a == False for a in trainFT)
        assert trainftv == True


        clf.set_trainable(True)

        trainFT = [l.trainable for l in clf.layers]
        trainftv = any(a == True for a in trainFT)
        assert trainftv == True
