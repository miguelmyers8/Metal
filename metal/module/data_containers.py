from .module import Module, is_module

# return modules container
class ModuleList(Module):
    def __init__(self, *layers):
        super(ModuleList, self).__init__()
        for c, l in enumerate(layers): setattr(self, l.__class__.__name__+str(c+1), l)

    def __iter__(self):
        for l in self.__dict__.values():
            if is_module(l):
                yield l

    def __repr__(self):
        return f'ModuleList{[ l.__class__.__name__+str(c+1) for c, l in enumerate(self)]}'

    def __getitem__(self, idx):
        return [ m for m in self.__dict__.values() if is_module(m) ][idx]

    def trainable_layers(self):
        for l in self.__dict__.values():
            if is_module(l):
                if l.trainable or l._params:
                    yield  l

    def _init_params(self):
        for c, i in enumerate(self):
            self[c]._init_params()


ModuleList.register()
