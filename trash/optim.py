"""
Optimizers go here
"""
from autograd.module import Module
from autograd.tensor import Tensor as tn
import numpy as np
class Optimizer(Module):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def save_params(self, module: Module) -> None:
        for i,j in enumerate(module.layers):
            if j.track_layer:
                self.param['w'+str(i+1)] = j.__dict__['w']
                self.param['b'+str(i+1)] = j.__dict__['b']

                self.grad['w'+str(i+1)]  = tn(j.__dict__['w'].grad.data)
                self.grad['b'+str(i+1)]  =  tn(j.__dict__['b'].grad.data)

    def zeroed_out(self,l, name=None):
        if name == 'Adam':
            for p in l.parameters():

                p.clear_s_corrected()

                p.clear_velocity_corrected()


        l.zero_grad()



class SGD(Optimizer):
    """docstring for SGD."""

    def __init__(self, lr: float = 0.01, beta: int = None):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.beta = beta

    def step(self, module: Module) -> None:


        if self.beta is not None:
            for l in module.layers:
                for parameter in l.parameters():
                    parameter.velocity = self.beta * parameter.velocity + (1-self.beta) * parameter.grad
                    parameter -= parameter.velocity * self.lr
                self.zeroed_out(l)

        else:
            for l in module.layers:
                for parameter in l.parameters():
                    parameter -= parameter.grad * self.lr
                self.zeroed_out(l)

class Adam(Optimizer):
    """docstring for Adam."""
    """
     vel_ -- Adam variable, moving average of the first gradient, python dictionary
     s_ -- Adam variable, moving average of the squared gradient, python dictionary
     learning_rate -- the learning rate, scalar.
     beta1 -- Exponential decay hyperparameter for the first moment estimates
     beta2 -- Exponential decay hyperparameter for the second moment estimates
     epsilon -- hyperparameter preventing division by zero in Adam updates
     t -- num of iteration
    """
    def __init__(self,lr=None,  beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8 ):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def step(self, module: Module,  t=0):
        for l in module.layers:
            for parameter in l.parameters():
                # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

                parameter.velocity = self.beta1 * parameter.velocity + (1-self.beta1) * parameter.grad


                # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
                parameter.velocity_corrected = parameter.velocity / (1-pow(self.beta1,t))


                # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
                parameter.s = self.beta2 * parameter.s + (1-self.beta2) * (parameter.grad*parameter.grad)


                # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
                parameter.s_corrected = parameter.s / (1-pow(self.beta2,t))


                # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
                parameter -=  self.lr * np.divide(parameter.velocity_corrected.data, np.sqrt(parameter.s_corrected.data ) + self.epsilon)


            self.zeroed_out(l, 'Adam')
