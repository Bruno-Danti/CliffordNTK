import numpy as np
from scipy.linalg.blas import dsyrk
from timeit import default_timer as timer


t0 = timer()

G: np.ndarray = np.loadtxt("../Part_2/G_matrix.csv",
                           float, delimiter= ',')

t1 = timer()

print(f"Time to load matrix: {t1 - t0}")
print(G.shape)
print(G[0:4, 0:5])

K: np.ndarray = dsyrk(alpha = 1.0, a = G, lower = False, overwrite_c = False)

t2 = timer()

print(f"Time to multiply G @ G.T: {t2 - t1}")
print(K.shape)
print(K[0:4, 0:4])