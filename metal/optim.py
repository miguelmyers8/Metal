"""
Optimizers go here
"""
from metal.module import Module

class GD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * self.lr
