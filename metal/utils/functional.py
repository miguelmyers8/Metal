import numpy as _np
from metal.autograd import numpy as np




#######################################################################
#                            Padding Utils                            #
#######################################################################


def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride, dilation=0):
    """
    Compute the padding necessary to ensure that convolving `X` with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.
    Parameters
    ----------
    X_shape : tuple of `(n_ex, in_rows, in_cols, in_ch)`
        Dimensions of the input volume. Padding is applied to `in_rows` and
        `in_cols`.
    out_dim : tuple of `(out_rows, out_cols)`
        The desired dimension of an output example after applying the
        convolution.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    Returns
    -------
    padding_dims : 4-tuple
        Padding dims for `X`. Organized as (left, right, up, down)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    pr = int((stride * (out_rows - 1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + _fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)

    # add asymmetric padding pixels to right / bottom
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    return (pr1, pr2, pc1, pc2)


def calc_pad_dims_1D(X_shape, l_out, kernel_width, stride, dilation=0, causal=False):
    """
    Compute the padding necessary to ensure that convolving `X` with a 1D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with length
    `l_out`.
    Parameters
    ----------
    X_shape : tuple of `(n_ex, l_in, in_ch)`
        Dimensions of the input volume. Padding is applied on either side of
        `l_in`.
    l_out : int
        The desired length an output example after applying the convolution.
    kernel_width : int
        The width of the 1D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    causal : bool
        Whether to compute the padding dims for a regular or causal
        convolution. If causal, padding is added only to the left side of the
        sequence. Default is False.
    Returns
    -------
    padding_dims : 2-tuple
        Padding dims for X. Organized as (left, right)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(l_out, int):
        raise ValueError("`l_out` must be of type int")

    if not isinstance(kernel_width, int):
        raise ValueError("`kernel_width` must be of type int")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fw = kernel_width
    n_ex, l_in, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fw = fw * (d + 1) - d
    total_pad = int((stride * (l_out - 1) + _fw - l_in))

    if not causal:
        pw = total_pad // 2
        l_out1 = int(1 + (l_in + 2 * pw - _fw) / stride)

        # add asymmetric padding pixels to right / bottom
        pw1, pw2 = pw, pw
        if l_out1 == l_out - 1:
            pw1, pw2 = pw, pw + 1
        elif l_out1 != l_out:
            raise AssertionError

    if causal:
        # if this is a causal convolution, only pad the left side of the
        # sequence
        pw1, pw2 = total_pad, 0
        l_out1 = int(1 + (l_in + total_pad - _fw) / stride)
        assert l_out1 == l_out

    if any(np.array([pw1, pw2]) < 0):
        raise ValueError("Padding cannot be less than 0. Got: {}".format((pw1, pw2)))
    return (pw1, pw2)


def pad1D(X, pad, kernel_width=None, stride=None, dilation=0):
    """
    Zero-pad a 3D input volume `X` along the second dimension.
    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
        Input volume. Padding is applied to `l_in`.
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple,
        specifies the number of padding columns to add on each side of the
        sequence.
    kernel_width : int
        The dimension of the 2D convolution kernel. Only relevant if p='same'
        or 'causal'. Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.
    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_seq, in_channels)`
        The padded output volume
    p : 2-tuple
        The number of 0-padded columns added to the (left, right) of the sequences
        in `X`.
    """
    p = pad
    if isinstance(p, int):
        p = (p, p)

    if isinstance(p, tuple):
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' or 'causal' convolution
    if p in ["same", "causal"] and kernel_width and stride:
        causal = p == "causal"
        p = calc_pad_dims_1D(
            X.shape, X.shape[1], kernel_width, stride, causal=causal, dilation=dilation
        )
        X_pad, p = pad1D(X, p)

    return X_pad, p


def pad2D(X, pad, kernel_shape=None, stride=None, dilation=0):
    """
    Zero-pad a 4D input volume `X` along the second and third dimensions.
    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume. Padding is applied to `in_rows` and `in_cols`.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        has the same dimensions as the input.  If 2-tuple, specifies the number
        of padding rows and colums to add *on both sides* of the rows/columns
        in `X`. If 4-tuple, specifies the number of rows/columns to add to the
        top, bottom, left, and right of the input volume.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel. Only relevant if p='same'.
        Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same'.
        Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same'.
        Default is 0.
    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_in_rows, padded_in_cols, in_channels)`
        The padded output volume.
    p : 4-tuple
        The number of 0-padded rows added to the (top, bottom, left, right) of
        `X`.
    """
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' convolution
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        X_pad, p = pad2D(X, p)
    return X_pad, p


def dilate(X, d):
    """
    Dilate the 4D volume `X` by `d`.
    Notes
    -----
    For a visual depiction of a dilated convolution, see [1].
    References
    ----------
    .. [1] Dumoulin & Visin (2016). "A guide to convolution arithmetic for deep
       learning." https://arxiv.org/pdf/1603.07285v1.pdf
    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume.
    d : int
        The number of 0-rows to insert between each adjacent row + column in `X`.
    Returns
    -------
    Xd : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The dilated array where
        .. math::
            \\text{out_rows}  &=  \\text{in_rows} + d(\\text{in_rows} - 1) \\\\
            \\text{out_cols}  &=  \\text{in_cols} + d (\\text{in_cols} - 1)
    """
    n_ex, in_rows, in_cols, n_in = X.shape
    r_ix = np.repeat(np.arange(1, in_rows), d)
    c_ix = np.repeat(np.arange(1, in_cols), d)
    Xd = np.insert(X, r_ix, 0, axis=1)
    Xd = np.insert(Xd, c_ix, 0, axis=2)
    return Xd


#######################################################################
#                     Convolution Arithmetic                          #
#######################################################################


def calc_conv_out_dims(X_shape, W_shape, stride=1, pad=0, dilation=0):
    """
    Compute the dimension of the output volume for the specified convolution.
    Parameters
    ----------
    X_shape : 3-tuple or 4-tuple
        The dimensions of the input volume to the convolution. If 3-tuple,
        entries are expected to be (`n_ex`, `in_length`, `in_ch`). If 4-tuple,
        entries are expected to be (`n_ex`, `in_rows`, `in_cols`, `in_ch`).
    weight_shape : 3-tuple or 4-tuple
        The dimensions of the weight volume for the convolution. If 3-tuple,
        entries are expected to be (`f_len`, `in_ch`, `out_ch`). If 4-tuple,
        entries are expected to be (`fr`, `fc`, `in_ch`, `out_ch`).
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple, specifies the
        number of padding columns to add on each side of the sequence. Default
        is 0.
    stride : int
        The stride for the convolution kernel. Default is 1.
    dilation : int
        The dilation of the convolution kernel. Default is 0.
    Returns
    -------
    out_dims : 3-tuple or 4-tuple
        The dimensions of the output volume. If 3-tuple, entries are (`n_ex`,
        `out_length`, `out_ch`). If 4-tuple, entries are (`n_ex`, `out_rows`,
        `out_cols`, `out_ch`).
    """
    dummy = np.zeros(X_shape)
    s, p, d = stride, pad, dilation
    if len(X_shape) == 3:
        _, p = pad1D(dummy, p)
        pw1, pw2 = p
        fw, in_ch, out_ch = W_shape
        n_ex, in_length, in_ch = X_shape

        _fw = fw * (d + 1) - d
        out_length = (in_length + pw1 + pw2 - _fw) // s + 1
        out_dims = (n_ex, out_length, out_ch)

    elif len(X_shape) == 4:
        _, p = pad2D(dummy, p)
        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = W_shape
        n_ex, in_rows, in_cols, in_ch = X_shape

        # adjust effective filter size to account for dilation
        _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d
        out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
        out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1
        out_dims = (n_ex, out_rows, out_cols, out_ch)
    else:
        raise ValueError("Unrecognized number of input dims: {}".format(len(X_shape)))
    return out_dims


#######################################################################
#                   Convolution Vectorization Utils                   #
#######################################################################


def _im2col_indices(X_shape, fr, fc, p, s, d=0):
    """
    Helper function that computes indices into X in prep for columnization in
    :func:`im2col`.
    Code extended from Andrej Karpathy's `im2col.py`
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

    # adjust effective filter size to account for dilation
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )

    # i1/j1 : row/col templates
    # i0/j0 : n. copies (len) and offsets (values) for row/col templates
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * (d + 1)
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * (d + 1)
    j1 = s * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (fr * fc * n_in, out_height * out_width)
    # j.shape = (fr * fc * n_in, out_height * out_width)
    # k.shape = (fr * fc * n_in, 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j


def im2col(X, W_shape, pad, stride, dilation=0):
    """
    Pads and rearrange overlapping windows of the input volume into column
    vectors, returning the concatenated padded vectors in a matrix `X_col`.
    Notes
    -----
    A NumPy reimagining of MATLAB's ``im2col`` 'sliding' function.
    Code extended from Andrej Karpathy's ``im2col.py``.
    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (not padded).
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights/kernels in the present convolutional
        layer.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in X. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    stride : int
        The stride of each convolution kernel
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    Returns
    -------
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape (Q, Z)
        The reshaped input volume where where:
        .. math::
            Q  &=  \\text{kernel_rows} \\times \\text{kernel_cols} \\times \\text{n_in} \\\\
            Z  &=  \\text{n_ex} \\times \\text{out_rows} \\times \\text{out_cols}
    """
    fr, fc, n_in, n_out = W_shape
    s, p, d = stride, pad, dilation
    n_ex, in_rows, in_cols, n_in = X.shape

    # zero-pad the input
    X_pad, p = pad2D(X, p, W_shape[:2], stride=s, dilation=d)
    pr1, pr2, pc1, pc2 = p

    # shuffle to have channels as the first dim
    X_pad = X_pad.transpose((0, 3, 1, 2))

    # get the indices for im2col
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, p, s, d)

    X_col = X_pad[:, k, i, j]
    X_col = X_col.transpose((1, 2, 0)).reshape(fr * fc * n_in, -1)
    return X_col, p


def col2im(X_col, X_shape, W_shape, pad, stride, dilation=0):
    """
    Take columns of a 2D matrix and rearrange them into the blocks/windows of
    a 4D image volume.
    Notes
    -----
    A NumPy reimagining of MATLAB's ``col2im`` 'sliding' function.
    Code extended from Andrej Karpathy's ``im2col.py``.
    Parameters
    ----------
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape `(Q, Z)`
        The columnized version of `X` (assumed to include padding)
    X_shape : 4-tuple containing `(n_ex, in_rows, in_cols, in_ch)`
        The original dimensions of `X` (not including padding)
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights in the present convolutional layer
    pad : 4-tuple of `(left, right, up, down)`
        Number of zero-padding rows/cols to add to `X`
    stride : int
        The stride of each convolution kernel
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    Returns
    -------
    img : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        The reshaped `X_col` input matrix
    """
    if not (isinstance(pad, tuple) and len(pad) == 4):
        raise TypeError("pad must be a 4-tuple, but got: {}".format(pad))

    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_ex, in_rows, in_cols, n_in = X_shape

    X_pad = np.zeros((n_ex, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, pad, s, d)

    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_ex)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return X_pad[:, :, pr1:pr2, pc1:pc2]
