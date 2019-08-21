import numpy as np
import time


from experimental.tensor import Tensor as tn
from experimental.src.functions import F_mat_mul as FF
#from experimental.lib.functions import F_mat_mul as FF

from cpp._mod1 import matmul

x = np.random.randn(4000,3000)
z = np.random.randn(3000,4000)

def t1(a1, b1):
    sum_ = tn(b1) @ tn(a1)
    return sum_

def t2(a1, b1):
    sum_ = b1 @ a1
    return sum_

s = time.time()
sum_ = matmul(z, x)
e = time.time()
print('cpp')
print(e-s)
print('')

s = time.time()
sum_ = t1(z, x)
e = time.time()
print('tensor/pythran')
print(e-s)
print('')

s = time.time()
sum_ = t2(z, x)
e = time.time()
print('numpy')
print(e-s)
print('')

s = time.time()
sum_ = FF(z, x)
e = time.time()
print('pythran')
print(e-s)
print('')
