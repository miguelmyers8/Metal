from ...autograd import numpy as np
import six
import itertools
from ...utils import as_tuple

def _im2col_indices(X_shape, k, s, outs, d=0):
    n_ex, n_in, in_rows, in_cols = X_shape
    out_rows, out_cols,_fr, _fc = outs
    fr, fc = k
    dy,dx = d
    sy,sx = s

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )
    # i1/j1 : row/col templates
    # i0/j0 : n. copies (len) and offsets (values) for row/col templates
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * dy
    i1 = sy * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * dx
    j1 = sx * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (fr * fc * n_in, out_height * out_width)
    # j.shape = (fr * fc * n_in, out_height * out_width)
    # k.shape = (fr * fc * n_in, 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def im2col_cpu(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    # img.shape = (n_ex,n_ch,nh,nw)
    # np.pad affects img (dim1,dim2,dim3,dim4)

    # dim3 padded (ph,ph)
    # dim3 gets (ph,ph)  a (vector of size nh added to top,vector of size nh added to bottom)
    # so now img is of shape(n_ex,n_ch,nh+ph+ph, nw)

    # dim4 padded (pw,pw)
    # dim4 gets (pw,pw) a (vector of size nw added to right, vector of size nw added to left)
    # so now img is of shape(n_ex,n_ch,nh+ph+ph, nw+pw+pw)
    img = np.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = np.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]
    return col

def col2im_cpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = np.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]

def im2col_nd_cpu(img, ksize, stride, pad, pval=0, cover_all=False, dilate=1):
    n, c = img.shape[0:2]       # (n, c, d_1, d_2, ..., d_N)
    dims = img.shape[2:]
    ndim = len(dims)
    dilate = as_tuple(dilate, ndim)
    assert ndim == len(ksize) == len(stride) == len(pad)
    outs = tuple(get_conv_outsize(d, k, s, p, cover_all, di)
                 for (d, k, s, p, di)
                 in zip(dims, ksize, stride, pad, dilate))
    assert all(out > 0 for out in outs), 'Output sizes should be positive.'

    # Pad around image.
    pad_width = ((0, 0), (0, 0)) + tuple(
        (p, p + s - 1) for (s, p) in zip(stride, pad))
    img = np.pad(img, pad_width, mode='constant', constant_values=(pval,))

    # Make patch array with which we will compute correlation with filter.
    # shape: (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
    shape = (n, c) + ksize + outs
    col = np.ndarray(shape, dtype=img.dtype)

    # Fill the patch array.
    colon = slice(None)

    for kxs in itertools.product(*[six.moves.range(k) for k in ksize]):
        # col[:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :]
        col_index = (colon, colon) + kxs + (colon,) * ndim
        # img[:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N]
        kx_dilate = tuple(kx * di for (kx, di) in zip(kxs, dilate))
        kx_lims = tuple(kx_di + s * out
                        for (kx_di, s, out) in zip(kx_dilate, stride, outs))
        img_index = (colon, colon) + tuple(
            slice(kx_di, kx_lim, s)
            for (kx_di, kx_lim, s) in zip(kx_dilate, kx_lims, stride))
        col[col_index] = img[img_index]
    return col


def col2im_nd_cpu(col, stride, pad, dims, dilate=1):
    n, c = col.shape[:2]  # (n, c, kx_1, ..., kx_N, out_1, ..., out_N)
    mid = (len(col.shape) - 2) // 2 + 2
    ksize = col.shape[2:mid]
    outs = col.shape[mid:]
    colon = slice(None)
    ndim = len(outs)
    dilate = as_tuple(dilate, ndim)
    assert len(ksize) == len(stride) == len(pad) == len(dims) == ndim

    # Image with padded size.
    img_shape = (n, c) + tuple(d + 2 * p + s - 1
                               for (d, p, s) in zip(dims, pad, stride))
    img = np.zeros(img_shape, dtype=col.dtype)
    for kxs in itertools.product(*[six.moves.range(k) for k in ksize]):
        # (:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N)
        kx_dilate = tuple(kx * di for (kx, di) in zip(kxs, dilate))
        kx_lims = tuple(kx_di + s * out
                        for (kx_di, s, out) in zip(kx_dilate, stride, outs))
        img_index = (colon, colon) + tuple(
            slice(kx_di, kx_lim, s)
            for (kx_di, kx_lim, s) in zip(kx_dilate, kx_lims, stride))
        # (:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :)
        col_index = (colon, colon) + kxs + (colon,) * len(outs)
        img[img_index] += col[col_index]

    # (:, :, p_1:d_1 + p_1, p_2:d_2 + p_2, ..., p_N:d_N + p_N]
    img_index = (colon, colon) + tuple(
        slice(p, d + p) for (p, d) in zip(pad, dims))
    return img[img_index]
