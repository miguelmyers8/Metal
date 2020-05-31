from ..module.module import Module
from ..module.data_containers import ModuleList

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = ModuleList(*tuple(layers))

    def __call__(self, x):
        for i, l in enumerate(self.layers):x = l(x)
        return x

    def __iter__(self): return iter(self.layers)

Sequential.register()
