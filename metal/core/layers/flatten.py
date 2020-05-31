import numpy as _np
from ..autograd import numpy as np
from ..autograd import Container
from ..autograd import primitive, defvjp

class FlattenFunction(object):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    @primitive
    def forward(self, inputs):
        x = inputs
        return x.reshape(x.shape[0], -1)

    def backward(ans,self,inputs):
        def grad(gy):
            gx = gy
            return _np.reshape(gx, inputs.shape)
        return grad

defvjp(FlattenFunction.forward, FlattenFunction.backward, argnums=(1,))
