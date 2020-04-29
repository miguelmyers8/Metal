def sgd_step(p, lr, **kwargs):
    p._value += -lr * p.grad
    return p
