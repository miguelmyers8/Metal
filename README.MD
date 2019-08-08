# Metal
Pytorch clone with a Auto differentiation framework for deep learning.

## Requirements
python 3.6<br>
xtensor-python<br>
xtensor<br>
xtensor-blas<br>
pybind11<br>


## Installation
Install this repo locally from top directory. <br />
$ pip install .



## Using Tensors
```python
    from autograd.tensor import Tensor as mt

    a = mt([1,2,3,4,5], requires_grad=True)
    b = mt([2,2,2,2,2], requires_grad=True)
    c = mt([5,4,3,2,1], requires_grad=True)
    d = a + b
    e = b + c
    f = d + e
    o = f.sum()
    o.backward()
    assert(b.grad.data.tolist() == np.array([2,2,2,2,2]).tolist())
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()
```


## Using Parameter
```python
    from autograd.parameter import Parameter as pr

    a = pr(inputs_=[1,2,3,4,5])
    b = pr(inputs_=[2,2,2,2,2])
    c = pr(inputs_=[5,4,3,2,1])
    d = a + b
    e = b + c
    f = d + e
    o = f.sum()
    o.backward()
    assert(b.grad.data.tolist() == np.array([2,2,2,2,2]).tolist())
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()


    a = pr(3,2)
    b = pr(1,2)
    c = pr(3,2)
    d = a + b
    e = b + c
    f = d + e
    o = f.sum()
    o.backward()
    assert(b.grad.data.tolist() == np.array([[6,6]]).tolist())
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()

```

## Working Example
```python
from autograd.tensor import Tensor as tn
from autograd.parameter import Parameter as pr
from autograd.flatten import Flatten as fl
from autograd.module import Module
from autograd.nn import Sequential
from autograd.linear import Linear
from autograd.act import Relu, Sigmoid
from autograd.loss import CEL
from autograd.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

train_x = fl().forward(train_x)
test_x = fl().forward(test_x)

train_y = tn(catnoncat.train_y())
test_y = tn(catnoncat.test_y())

optimizer = SGD(lr=.0075)

mod2 = Sequential([
         Linear(50,12288),
            Relu(),
         Linear(20,50),
            Relu(),
         Linear(10,20),
            Relu(),
         Linear(1,10),
            Sigmoid()])

costs = []

for epoch in range(7000):

    epoch_loss = 0.0

    out = mod2.forward(train_x)
    cout = CEL(out,train_y)
    loss = cout.sum()

    loss.backward()
    epoch_loss += loss.data


    optimizer.step(mod2)


    if epoch % 100 == 0:
        print(epoch, epoch_loss)
    if epoch % 100 == 0:
        costs.append(epoch_loss)


# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.show()


mod2.predict(test_x,test_y)

->Accuracy: 0.76
```
