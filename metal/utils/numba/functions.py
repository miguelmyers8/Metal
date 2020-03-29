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
