import _autograd._math as am
import numpy as np
import time


x=np.random.randn(4000,3000)
z=np.random.randn(3000,4000)
k=np.random.randn(4000,5000)


start = time.time()
for i in range(5):
    c = x@z
    c@k
end = time.time()
print('np: ',end - start)

start = time.time()
for i in range(5):
    c  = am.matmul(x,z)
    am.matmul(c,k)
end = time.time()
print('xt: ',end - start)
