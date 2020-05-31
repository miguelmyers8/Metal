from ....autograd import numpy as anp

def logsumexp(x):
    m = x.max(axis=-1, keepdims=True)[0]
    return m + anp.log(anp.exp((x-m[:,None])).sum(axis=-1, keepdims=True))

def log_softmax(x): return x - logsumexp(x)

def nll(inputs, target): return -inputs[range(target.shape[0]), target].mean()

def cross_entropy_loss(y, ypred): return nll(log_softmax(y), ypred)
