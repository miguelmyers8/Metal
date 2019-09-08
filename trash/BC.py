
import numpy as np
import metal
from metal.tensor import Tensor as tn
from metal.parameter import Parameter as pr
from metal.flatten import Flatten as fl
from metal.module import Module
from metal.nn import Sequential
from metal.linear import Linear
from metal.act import Relu, Sigmoid
from metal.loss import CEL
from metal.optim import GD
import matplotlib.pyplot as plt



from Dataset1 import catnoncat

train_x = catnoncat.train_x()
test_x = catnoncat.test_img()


train_x = fl().forward(train_x)
test_x = fl().forward(test_x)

train_y = tn(catnoncat.train_y())
test_y = tn(catnoncat.test_y())

mod2 = Sequential([
         Linear(50,12288),
            Relu(),
         Linear(20,50),
            Relu(),
         Linear(10,20),
            Relu(),
         Linear(1,10),
            Sigmoid()])

optimizer = GD(lr=.0075)


costs = []

for epoch in range(7_000):

    epoch_loss = 0.0

    for l in mod2.layers:
        l.zero_grad()

    out = mod2.forward(train_x)
    cout = CEL(out,train_y)
    loss = cout.sum()

    loss.backward()
    epoch_loss += loss.data

    for l in mod2.layers:
        optimizer.step(l)


    if epoch % 100 == 0:
        print(epoch, epoch_loss)
    if epoch % 100 == 0:
        costs.append(epoch_loss)


# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.show()

pred = mod2.predict(test_x,test_y)

print(pred)
