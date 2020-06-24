# Metal
A toy deep learning framework. This repo consist of my multiple deep learning tutorial and github repos group together to form this project.

## Requirements
Python 3.6+<br>
Autograd [clone from here](https://github.com/miguelmyers8/autodiff)

## Installation
Install this repo locally from top directory.<br/>
$ pip install -e .


## Example
```Python
from metal.autograd import Container
from metal.autograd import numpy as anp
from metal.autograd import no_grad
from metal.layers.dense import Dense
from metal.layers.conv2D import Conv2D
from metal.layers.flatten import Flatten
from metal.core.kernels.functions.loss import cross_entropy_loss
from metal.core.utils import accuracy
from metal.models.sequential import Sequential
from metal.module.data_containers import ModuleList
from metal.core.steppers import sgd_step
from metal.core.samplers import *
from metal.core._01_base import *
from metal.core import Optimizer
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

def get_model():
    model = Sequential([Flatten(), Dense(784,50,act_fn='relu'),Dense(50,10)])
    model._init_params()
    return model, Optimizer(model.parameters(),sgd_step,lr=0.5)

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds,sampler=RandomSampler(train_ds),batch_size=64,collate_fn=collate),
            DataLoader(valid_ds,sampler=SequentialSampler(valid_ds),batch_size=64*2,collate_fn=collate))    

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred._value, yb)
                tot_acc  += accuracy (pred._value, yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_,x_test_ = (x_train/255.0).reshape(-1,1,28,28).astype('float32'), (x_test/255.0).reshape(-1,1,28,28).astype('float32')
train,test = Dataset(x_train_,y_train),Dataset(x_test_,y_test)
train_dll, test_dll = get_dls(train,test,64)
mod, opt = get_model()

l,a = fit(5,mod,cross_entropy_loss,opt,train_dll,test_dll)

0 0.1790674288250223 0.9426424050632911
1 0.13143482919829555 0.9588607594936709
2 0.11844033200906802 0.9647943037974683
3 0.09235251610037647 0.9709256329113924
4 0.10407011588163013 0.9676621835443038       
```

## Learned resources
[joelgrus/autograd](https://github.com/joelgrus/autograd)<br>
[eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)<br>
[ddbourgin/numpy-ml](https://github.com/ddbourgin/numpy-ml)<br>
[cs231](http://cs231n.github.io/)<br>
[deeplearning.ai](https://www.deeplearning.ai/)<br>
[fastai](https://github.com/fastai/course-v3/blob/master/nbs/dl2)<br>
[chainer](https://chainer.org/)<br>

### Todo
- [ ] Finish Unittest
- [ ] Integrate cupy for gpu computation into autograd
- [ ] Integrate viz. saliency-maps
- [ ] data augmentation
