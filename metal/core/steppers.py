def sgd_step(p, lr, **kwargs):
    p._value += -lr * p.grad
    return p

def weight_decay(p, lr, **kwargs):
    p._value *= 1 - lr*wd
    return p
weight_decay._defaults = dict(wd=0.)

def l2_reg(p, lr, wd, **kwargs):
    p.grad += wd * p._value
    return p
l2_reg._defaults = dict(wd=0.)
