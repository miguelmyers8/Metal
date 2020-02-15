from numba import jit
import numpy as np
from metal.utils.utils import im2col

@jit(nopython=True)
def _pool2D(n_ex,out_rows,out_cols, out_ch,X_pad,s,fr,fc,Y,mode):

    for m in range(n_ex):
        for i in range(out_rows):
            for j in range(out_cols):
                for c in range(out_ch):
                    # calculate window boundaries, incorporating stride
                    i0, i1 = i * s, (i * s) + fr
                    j0, j1 = j * s, (j * s) + fc

                    xi = X_pad[m, i0:i1, j0:j1, c]

                    if mode == "max":
                        Y[m, i, j, c] = np.max(xi)
                    elif mode == "average":
                        Y[m, i, j, c] = np.mean(xi)
    return Y


@jit(nopython=True)
def _pool2D_backward(n_ex,out_row,out_col,out_ch,fr,fc,s,mode,dX,X,dy):
    for m in range(n_ex):
        for i in range(out_row):
            for j in range(out_col):
                for c in range(out_ch):
                    # calculate window boundaries, incorporating stride
                    i0, i1 = i * s, (i * s) + fr
                    j0, j1 = j * s, (j * s) + fc

                    if mode == "max":
                        xi = X[m, i0:i1, j0:j1, c]

                        # enforce that the mask can only consist of a
                        # single `True` entry, even if multiple entries in
                        # xi are equal to max(xi)
                        mask = np.zeros_like(xi)>1
                        x, y = ((xi == np.max(xi))[0])*1
                        mask[x, y] = True

                        dX[m, i0:i1, j0:j1, c] += mask * dy[m, i, j, c]
                    elif mode == "average":
                        frame = np.ones((fr, fc),dtype=np.float32) * dy[m, i, j, c]
                        dX[m, i0:i1, j0:j1, c] += frame / np.prod(np.array((fr, fc)))
    return dX



#Y = _pool2D_V1_(X,self.kernel_shape,out_rows, out_cols,nc_in,self.stride,self.mode,retain_derived)
def _pool2D_V1_(a_prev,pool_size,n_h, n_w,n_c,stride,mode,retain_derived, cache):
    batch_size = a_prev.shape[0]
    a = np.zeros((batch_size, n_h, n_w, n_c))

    # Pool
    for i in range(n_h):
        v_start = i * stride
        v_end = v_start + pool_size[0]

        for j in range(n_w):
            h_start = j * stride
            h_end = h_start + pool_size[0]

            if mode == 'max':
                a_prev_slice = a_prev[:, v_start:v_end, h_start:h_end, :]

                if retain_derived:
                    # Cache for backward pass
                    cache = cache_max_mask(a_prev_slice, (i, j), cache)

                a[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))

            elif mode == 'average':
                a[:, i, j, :] = np.mean(a_prev[:, v_start:v_end, h_start:h_end, :], axis=(1, 2))

            else:
                raise NotImplementedError("Invalid type of pooling")

    return a, cache


def cache_max_mask(x, ij, cache):
    mask = np.zeros_like(x)

    # This would be like doing idx = np.argmax(x, axis=(1,2)) if that was possible
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
    idx = np.argmax(reshaped_x, axis=1)

    ax1, ax2 = np.indices((x.shape[0], x.shape[3]))
    mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1
    cache[ij] = mask
    return cache


def _pool2D_V1_backward(da,cache,n_hp,n_wp,n_cp,stride,pool_size,mode,n_h,n_w):
    a_prev = cache['a_prev']
    batch_size = a_prev.shape[0]
    da_prev = np.zeros((batch_size, n_hp, n_wp, n_cp))
    # 'Pool' back
    for i in range(n_h):
        v_start = i * stride
        v_end = v_start + pool_size[0]

        for j in range(n_w):
            h_start = j * stride
            h_end = h_start + pool_size[0]

            if mode == 'max':

                da_prev[:, v_start:v_end, h_start:h_end, :] += da[:, i:i+1, j:j+1, :] * cache[(i, j)]

            elif mode == 'average':
                # Distribute the average value back
                mean_value = np.copy(da[:, i:i+1, j:j+1, :])
                mean_value[:, :, :, np.arange(mean_value.shape[-1])] /= (pool_size * pool_size[0])
                da_prev[:, v_start:v_end, h_start:h_end, :] += mean_value

            else:
                raise NotImplementedError("Invalid type of pooling")

    return da_prev
