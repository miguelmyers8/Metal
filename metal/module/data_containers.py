from .module import Module

# return updateable modules from a list or all
class ModuleList(Module):
    def __init__(self, layers=None):
        super(ModuleList, self).__init__()
        self.layers = layers

    def __iter__(self):
        for l in self.layers:
            yield l

    def trainable_layers(self):
        for l in self:
            if l.trainable or l._params:
                yield  l
