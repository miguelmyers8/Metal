import six
from metal.autograd import numpy as np
from metal.autograd import primitive, defvjp
from ..kernels.functions.functional import im2col_nd_cpu, col2im_nd_cpu
from metal.utils.utils import _pair, parse_kwargs, as_tuple
import functools
from operator import mul


class MaxPoolingND():
    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True,return_indices=False):
        super(MaxPoolingND, self).__init__()
        self.ndim = ndim
        self.ksize = as_tuple(ksize, ndim)
        self.stride = as_tuple(stride, ndim)
        self.pad = as_tuple(pad, ndim)
        self.cover_all = cover_all
        self.return_indices = return_indices

    @primitive
    def forward_cpu(self, x):
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all

        in_shape = x.shape
        in_dtype = x.dtype

        col = im2col_nd_cpu(x, ksize, stride, pad,pval=-float('inf'),cover_all=cover_all)
        n, c = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        # (n, c, k_1 * k_2 * ... * k_N, out_1, out_2, ..., out_N)
        col_shape = (n, c) + (functools.reduce(mul, ksize),) + outs
        col = col.reshape(col_shape)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        y = col.max(axis=2)

        self._in_shape = in_shape
        self._in_dtype = in_dtype
        self.indexes = col.argmax(axis=2)
        return y

    def backward(ans, self, x):
        def grad (gy):
            func = self
            ndim = func.ndim
            ksize = func.ksize
            stride = func.stride
            pad = func.pad
            in_shape = func._in_shape
            in_dtype = func._in_dtype
            indexes = func.indexes
            n, c = gy.shape[:2]
            outs = gy.shape[2:]
            dims = in_shape[2:]
            prod_outs = functools.reduce(mul, outs)
            prod_ksize = functools.reduce(mul, ksize)
            gcol = np.zeros(n * c * prod_outs * prod_ksize, dtype=in_dtype)
            indexes = (indexes.flatten() + np.arange(0, indexes.size * prod_ksize, prod_ksize))
            gcol[indexes] = gy.ravel()
            gcol_shape = (n, c) + outs + ksize
            gcol = gcol.reshape(gcol_shape)
            for i in six.moves.range(ndim):
                gcol = np.swapaxes(gcol, 2 + i, ndim + 2 + i)
            gx = col2im_nd_cpu(gcol, stride, pad, dims)
            return gx
        return grad

defvjp(MaxPoolingND.forward_cpu, MaxPoolingND.backward, argnums=(1,))


def max_pooling_nd(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    ndim = len(x.shape[2:])
    func = MaxPoolingND(ndim, ksize, stride, pad, cover_all, return_indices)
    return func.forward_cpu((x,)[0])
