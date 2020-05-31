import autograd.numpy as numpy
from autograd.numpy.container import container, VJPNode
import numpy as _np
from autograd.extend import defjvp,defvjp,vspace
from autograd.util import func
from autograd.numpy.numpy_vjps import untake
from autograd.numpy.numpy_vjps import wrapped_reshape
from autograd.tracer import primitive, is_container, no_grad, using_config, test_mode
from autograd.core import backward as _backward
from autograd.numpy import numpy_wrapper as anp, primitive

nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']

class container_mateclass(type):
    def __new__(self, name,base, dic):
        cls = type.__new__(container_mateclass,name, base, dic)
        cls.register(_np.ndarray)
        for type_ in [float, _np.float64, _np.float32, _np.float16,
              complex, _np.complex64, _np.complex128]:
               cls.register(type_)
        for method_name in nondiff_methods + diff_methods:
            setattr(cls, method_name, anp.__dict__[method_name])
        setattr(cls, 'flatten', anp.__dict__['ravel'])
        defvjp(func(cls.__getitem__), lambda ans, A, idx: lambda g: untake(g, idx, vspace(A)))
        defjvp(func(cls.__getitem__), 'same')
        defjvp(untake, 'same')
        setattr(cls, 'reshape', wrapped_reshape)
        return cls

class Container_(container, metaclass=container_mateclass):
    def __init__(self, _value,requires_grad=False, _node=None):
        super(Container_,self).__init__(_value,requires_grad,_node)

def Container(val,requires_grad=False):
    return Container_(val,requires_grad=requires_grad,_node=VJPNode.new_root())
