from ..layers.conv2D import Conv2D
from ..layers.dense import Dense
from ..layers.maxpool2d import MaxPool2D
from ..module.module import Module
from ..layers.dropout import Dropout
from ..layers.flatten import Flatten
from ..autograd import no_grad
from metal.models.pre_trained_weights.vggloader import vggweights,prepare

import numpy as np

def zeros(x):
    return np.zeros(x)

def _max_pooling_2d(x):
    return MaxPool2D(stride=(2,2), ksize=(2,2), pad=(0,0))(x)

def _flatten(x):
    return Flatten()(x)

def _dropout(x):
    return Dropout()(x)

class VGG(Module):
    def __init__(self, pretrained_model=True, n_layers=16):
        super().__init__()
        kwargs = {"init":zeros}
        self.labels = vggweights[1]
        self.conv1_1 = Conv2D(   3, 64, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv1_2 = Conv2D(  64, 64, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv2_1 = Conv2D( 64, 128, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv2_2 = Conv2D(128, 128, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv3_1 = Conv2D(128, 256, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv3_2 = Conv2D(256, 256, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv3_3 = Conv2D(256, 256, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv4_1 = Conv2D(256, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv4_2 = Conv2D(512, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv4_3 = Conv2D(512, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv5_1 = Conv2D(512, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv5_2 = Conv2D(512, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.conv5_3 = Conv2D(512, 512, 3, 1, 1, act_fn='relu',**kwargs)
        self.fc6 = Dense(512 * 7 * 7, 4096, **kwargs)
        self.fc7 = Dense(4096, 4096, **kwargs)
        self.fc8 = Dense(4096, 1000, **kwargs)

        if n_layers == 19:
            self.conv3_4 = Conv2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_4 = Conv2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_4 = Conv2D(512, 512, 3, 1, 1, **kwargs)

        if pretrained_model:
            self._init_params()
            if n_layers == 16:
                self.load_weights(weights=vggweights[0])


class VGG16(VGG):

    def __init__(self, pretrained_model=True):
        super(VGG16, self).__init__(pretrained_model, 16)

    @property
    def vgg_layers(self):
        layers = [
                 self.conv1_1,self.conv1_2,_max_pooling_2d,
                 self.conv2_1,self.conv2_2,_max_pooling_2d,
                 self.conv3_1,self.conv3_2,   self.conv3_3,
                 _max_pooling_2d,self.conv4_1,self.conv4_2,
                 self.conv4_3,_max_pooling_2d,self.conv5_1,
                 self.conv5_2,self.conv5_3,_max_pooling_2d,
                 _flatten,self.fc6,_dropout,self.fc7,
                 _dropout, self.fc8
                 ]
        return layers

    def forward(self,x):
        for i, l in enumerate(self.vgg_layers): x = l(x)
        return x

def vgg16_function(X,model):
    assert isinstance(model,VGG16)
    model.eval()
    with no_grad():
        vgg16out = model(X)
    return model.labels[vgg16out._value.argmax()], vgg16out
