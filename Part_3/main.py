import numpy as np, sys
from scipy.linalg.blas import dsyrk

n_images = int(sys.argv[1])
G: np.ndarray = np.loadtxt(sys.argv[2], float, delimiter=',', max_rows=n_images)
K: np.ndarray = dsyrk(alpha = 1.0, a = G, lower = False)

np.savetxt(sys.argv[3], K, delimiter= ",")