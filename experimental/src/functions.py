import numpy as np

#pythran export F_mat_mul(float32[][], float32[][])
#pythran export F_mat_mul(float32[][][], float32[][][])
#pythran export F_mat_mul(float64[:,:], float64[:,:])
#pythran export F_mat_mul(float64[:,:,:], float64[:,:,:])
def F_mat_mul(a,b):
    return b

#pythran export D1_mat_mul(str:float32[][][] dict)
#pythran export D1_mat_mul(str:float64[:,:,:] dict)
def D1_mat_mul(a):
    return a
