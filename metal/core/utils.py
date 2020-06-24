from typing import *
from .autograd import numpy as anp
import numpy as np
import numbers


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

def as_tuple(x, n):
    if hasattr(x, '__getitem__'):
        assert len(x) == n
        return tuple(x)
    return (x,) * n

def check_unexpected_kwargs(kwargs, **unexpected):
    for key, message in unexpected.items():
        if key in kwargs:
            raise ValueError(message)


def parse_kwargs(kwargs, *name_and_values, **unexpected):
    values = [kwargs.pop(name, default_value)
              for name, default_value in name_and_values]
    if kwargs:
        check_unexpected_kwargs(kwargs, **unexpected)
        caller = inspect.stack()[1]
        args = ', '.join(repr(arg) for arg in sorted(kwargs.keys()))
        message = caller[3] + \
            '() got unexpected keyword argument(s) {}'.format(args)
        raise TypeError(message)
    return tuple(values)

def accuracy(out, yb): return (anp.argmax(out, axis=1)==yb).mean()

def get_top_classes(x,labels,num_classes=4):
    ind = list(np.argpartition(x[1][0]._value, -num_classes)[-num_classes:])
    ind.reverse()
    return {labels.get(i):x[1][0]._value[ind][c] for c,i in enumerate(ind)}
