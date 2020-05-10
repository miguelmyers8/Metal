import inspect


def size_of_shape(shape):
    size = 1
    for i in shape:
        size *= i
    # should not return long in Python 2
    return int(size)

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
