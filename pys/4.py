import numpy as np, numba as nb
from scipy.linalg.blas import dsyrk
from timeit import default_timer as timer

N = 30000
K = 252

G_mat = np.random.random((N, K))

start_time = timer()
#K_mat = G_mat @ G_mat.T
K_mat = dsyrk(alpha = 1.0, a = G_mat, lower = False, overwrite_c = False)
end_time = timer()
print(f"Np time to multiply vectorized matrix: {end_time - start_time}")

import sys
print(f"{sys.getsizeof(K_mat) / 10**9} GB")