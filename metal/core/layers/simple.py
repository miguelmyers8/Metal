from .conv2d import Convolution2DFunction, parse_kwargs
from .dense import DenseFunction

def max_pooling_nd(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    ndim = len(x.shape[2:])
    func = MaxPoolingND(ndim, ksize, stride, pad, cover_all, return_indices)
    return func.forward_cpu((x,)[0])

def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if len(x.shape[2:]) != 2:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 2. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)

def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, **kwargs):
    dilate, groups, cudnn_fast = parse_kwargs(kwargs, ('dilate', 1), ('groups', 1), ('cudnn_fast', False))
    fnode = Convolution2DFunction(stride, pad, cover_all, dilate=dilate, groups=groups, cudnn_fast=cudnn_fast)
    return fnode._forward_cpu_core(x,W,b)

def dense(x, W, b):
    return DenseFunction().forward_cpu(x,W,b)
