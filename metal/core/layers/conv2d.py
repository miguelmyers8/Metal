import six
from ..autograd import numpy as np
from ..autograd import primitive, defvjp
from ..kernels.functions.functional import col2im_cpu, im2col_cpu, get_conv_outsize, _im2col_indices, im2col_nd_cpu, col2im_nd_cpu
from metal.utils.utils import _pair, parse_kwargs, as_tuple
import functools
from operator import mul


class Convolution2DFunction():
    def __init__(self, stride=1, pad=0, cover_all=False, **kwargs):
        dilate, groups, cudnn_fast = parse_kwargs(kwargs, ('dilate', 1), ('groups', 1), ('cudnn_fast', False))
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all
        self.dy, self.dx = _pair(dilate)
        self.groups = groups
        self.cudnn_fast = cudnn_fast

    @primitive
    def _forward_cpu_core(self, x, W, b):
        kh, kw = W.shape[2:]
        col = im2col_cpu( x, kh, kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y = y+b
        y = np.rollaxis(y, 3, 1)
        return y

    def _get_out_size(self, x_shape, w_shape):
        _, _, kh, kw = w_shape
        _, _, h, w = x_shape

        out_h = get_conv_outsize(
            h, kh, self.sy, self.ph, cover_all=self.cover_all, d=self.dy)[0]
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = get_conv_outsize(
            w, kw, self.sx, self.pw, cover_all=self.cover_all, d=self.dx)[0]
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return out_h, out_w

    def backwardX(ans, self, x, W, b):
        def grad(gy):
            _, _, xh, xw = x.shape
            gcol = np.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
            gcol = np.rollaxis(gcol, 3)
            y = col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, xh,xw, dy=self.dy, dx=self.dx)
            if b is not None:
                y = y + b.reshape((1, b.size, 1, 1))
            return y
        return grad

    def backwardW(ans,self,x,W,b):
        def grad(gy):
            kh, kw = W.shape[2::]
            col = im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all, dy=self.dy, dx=self.dx)
            gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)
            return gW
        return grad

    def backwardb(ans,self,x,W,b):
        def grad(gy):
            axis = (0, 2, 3)
            gb = np.sum(gy, axis=axis)
            return gb
        return grad


defvjp(Convolution2DFunction._forward_cpu_core, Convolution2DFunction.backwardX, Convolution2DFunction.backwardW, Convolution2DFunction.backwardb, argnums=(1,2,3))
