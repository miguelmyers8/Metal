"""
Optimizers go here
"""
from autograd.module import Module
from autograd.tensor import Tensor as tn

class GD(Module):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def save_params(self, module: Module) -> None:
        for i,j in enumerate(module.layers):
            if j.track_layer:
                self.param['w'+str(i+1)] = j.__dict__['w']
                self.param['b'+str(i+1)] = j.__dict__['b']

                self.grad['w'+str(i+1)]  = tn(j.__dict__['w'].grad.data)
                self.grad['b'+str(i+1)]  =  tn(j.__dict__['b'].grad.data)

    def zeroed_out(self,l):
        l.zero_grad()

    def step(self, module: Module) -> None:
        for l in module.layers:
            for parameter, _ in l.parameters():
                parameter -= parameter.grad * self.lr
            self.zeroed_out(l)
