from ..autograd import numpy as np
from ..autograd import primitive, defvjp


class DenseFunction(object):
    @primitive
    def forward_cpu(self,x,w,b):
        z = x @ w.T + b
        return z

    def backwardx(ans,self,x,w,b):
        def grad (gy):
            gx = gy.dot(w).astype(gy.dtype, copy=False)
            return gx
        return grad

    def backwardw(ans,self,x,w,b):
        def grad (gy):
            gw = gy.T.dot(x).astype(w.dtype, copy=False)
            return gw
        return grad

    def backwardb(ans,self,x,w,b):
        def grad (gy):
            gb = np.sum(gy, axis=0)
            return gb
        return grad
defvjp(DenseFunction.forward_cpu, DenseFunction.backwardx,DenseFunction.backwardw,DenseFunction.backwardb, argnums=(1,2,3))
