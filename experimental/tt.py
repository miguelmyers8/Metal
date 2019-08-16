import numpy as np
from experimental import m2 #a1
from experimental import mym #a
import time
np.random.seed(1)

x = np.random.randn(4000,300)
zz = np.random.randn(300,4000)
class test:
    def __init__(self):
        pass

    def mul(self,x1,zz1,f1):
        return mym.a(x1,zz1,f1)


def z(x,zz,f):
    for i in range(f):
        c = zz @ x
    return c

s = time.time()
test().mul(x,zz,80)
e= time.time()
print(e-s)

s = time.time()
z(x,zz,80)
e= time.time()
print(e-s)
