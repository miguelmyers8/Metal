import numpy as np
from ..autograd import numpy as anp
from ..autograd import primitive, defvjp

class DropoutFunction():
    def __init__(self, dropout_ratio, mask=None, return_mask=False):
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError('dropout_ratio must be in the range [0, 1)')
        self.dropout_ratio = dropout_ratio
        self.mask = mask
        self.return_mask = return_mask
        self._use_cudnn = False

    @primitive
    def forward_cpu(self, x):
        if self.mask is not None:
            y = x * self.mask
        else:
            scale = x.dtype.type(1. / (1 - self.dropout_ratio))
            flag = np.random.rand(*x.shape) >= self.dropout_ratio
            self.mask = scale * flag
            y = x * self.mask
        return y

    def backward_dropout(ans,self,x):
        def grad(gy):
            y = gy * self.mask
            return y
        return grad

defvjp(DropoutFunction.forward_cpu,DropoutFunction.backward_dropout, argnums=(1,))
