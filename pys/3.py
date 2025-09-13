import numpy as np, numba as nb
from timeit import default_timer as timer

@nb.njit
def dot_prod(a, b):
    sum = 0
    for i in range(a.shape[0]):
        sum += a[i] * b[i]
    return sum

N = 252
test_vec1 = np.random.random(N)
test_vec2 = np.random.random(N)

dot_prod(test_vec1, test_vec2)
start_time = timer()
dot_prod(test_vec1, test_vec2)
#test_vec1.dot(test_vec2)
end_time = timer()
print(f"Np time to product cached {N}-long vectors: {end_time - start_time}")