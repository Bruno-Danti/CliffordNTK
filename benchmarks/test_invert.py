import numpy as np
from timeit import default_timer as timer

N = 10000

K = np.random.random((N, N))

t0 = timer()

K_inv = np.linalg.inv(K)

t1 = timer()

print(t1 - t0)