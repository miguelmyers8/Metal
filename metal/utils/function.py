from metal.autograd import numpy as np
from .functional import pad2D, im2col, col2im, dilate, calc_pad_dims_2D

#_pool2D(x, pool_size, strides, padding, pool_mode,data_format)
def _pool2D(x, pool_size, strides, padding, pool_mode,data_format):
    if data_format == 'channels_last':
        if x.ndim == 3:
            x = np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            x = np.transpose(x, (0, 3, 1, 2))
        else:
            x = np.transpose(x, (0, 4, 1, 2, 3))

    if padding == 'same':
        pad = [(0, 0), (0, 0)] + [(s // 2, s // 2) for s in pool_size]
        x = np.pad(x, pad, 'constant', constant_values=-np.inf)

    # indexing trick
    x = np.pad(x, [(0, 0), (0, 0)] + [(0, 1) for _ in pool_size],
               'constant', constant_values=0)

    if x.ndim == 3:
        y = [x[:, :, k:k1:strides[0]]
             for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0))]
    elif x.ndim == 4:
        y = []
        for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0)):
            for (l, l1) in zip(range(pool_size[1]), range(-pool_size[1], 0)):
                y.append(x[:, :, k:k1:strides[0], l:l1:strides[1]])
    else:
        y = []
        for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0)):
            for (l, l1) in zip(range(pool_size[1]), range(-pool_size[1], 0)):
                for (m, m1) in zip(range(pool_size[2]), range(-pool_size[2], 0)):
                    y.append(x[:,
                               :,
                               k:k1:strides[0],
                               l:l1:strides[1],
                               m:m1:strides[2]])
    y = np.stack(y, axis=-1)
    if pool_mode == 'avg':
        y = np.mean(np.ma.masked_invalid(y), axis=-1).data
    elif pool_mode == 'max':
        y = np.max(y, axis=-1)

    if data_format == 'channels_last':
        if y.ndim == 3:
            y = np.transpose(y, (0, 2, 1))
        elif y.ndim == 4:
            y = np.transpose(y, (0, 2, 3, 1))
        else:
            y = np.transpose(y, (0, 2, 3, 4, 1))

    return y


def conv2D(X, W, stride, pad, dilation=0):
    s, d = stride, dilation
    _, p = pad2D(X, pad, W.shape[:2], s, dilation=dilation)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_ex, in_rows, in_cols, in_ch = X.shape

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # compute the dimensions of the convolution output
    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    # convert X and W into the appropriate 2D matrices and take their product
    X_col, _ = im2col(X, W.shape, p, s, d)
    W_col = W.transpose((3, 2, 0, 1)).reshape(out_ch, -1)

    Z = (W_col @ X_col).reshape(out_ch, out_rows, out_cols, n_ex).transpose((3, 1, 2, 0))

    return Z


def conv1D(X, W, stride, pad, dilation=0):
    _, p = pad1D(X, pad, W.shape[0], stride, dilation=dilation)

    # add a row dimension to X to permit us to use im2col/col2im
    X2D = np.expand_dims(X, axis=1)
    W2D = np.expand_dims(W, axis=0)
    p2D = (0, 0, p[0], p[1])
    Z2D = conv2D(X2D, W2D, stride, p2D, dilation)

    # drop the row dimension
    return np.squeeze(Z2D, axis=1)



def deconv2D_naive(X, W, stride, pad, dilation=0):
    if stride > 1:
        X = dilate(X, stride - 1)
        stride = 1

    # pad the input
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=stride, dilation=dilation)

    n_ex, in_rows, in_cols, n_in = X_pad.shape
    fr, fc, n_in, n_out = W.shape
    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = p

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # compute deconvolution output dims
    out_rows = s * (in_rows - 1) - pr1 - pr2 + _fr
    out_cols = s * (in_cols - 1) - pc1 - pc2 + _fc
    out_dim = (out_rows, out_cols)

    # add additional padding to achieve the target output dim
    _p = calc_pad_dims_2D(X_pad.shape, out_dim, W.shape[:2], s, d)
    X_pad, pad = pad2D(X_pad, _p, W.shape[:2], stride=s, dilation=dilation)

    # perform the forward convolution using the flipped weight matrix (note
    # we set pad to 0, since we've already added padding)
    Z = conv2D(X_pad, np.rot90(W, 2), s, 0, d)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return Z[:, pr1:pr2, pc1:pc2, :]


def dense( X,W,b):
    Z = X @ W + b
    return  Z

def flatten(X,keep_dim):
    if keep_dim == -1:
        return X.flatten().reshape(1, -1)

    rs = (X.shape[0], -1) if keep_dim == "first" else (-1, X.shape[-1])
    return X.reshape(*rs)

def softmax(X, dim):
    """Actual computation of softmax forward pass"""
    # center data to avoid overflow
    e_X = np.exp(X - np.max(X, axis=dim, keepdims=True))
    return e_X / e_X.sum(axis=dim, keepdims=True)
