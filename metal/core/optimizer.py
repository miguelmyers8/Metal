from .extend import compose, listify
from .callbacks import LR_Finder, Recorder
from functools import partial

def maybe_update(os, dest, f):
    for o in os:
        for k,v in f(o).items():
            if k not in dest: dest[k] = v

def get_defaults(d): return getattr(d,'_defaults',{})


class Optimizer(object):
    def __init__(self, params, steppers, **defaults):
        # might be a generator
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        for p, hyper in self.grad_params():
            p.zero_grad()

    def step(self):
        for p, hyper in self.grad_params(): compose(p, self.steppers, **hyper)

sgd_opt = partial(Optimizer, steppers=[])
