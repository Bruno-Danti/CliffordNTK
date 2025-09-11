from timeit import default_timer as timer
from numba import njit

@njit
def empty_function():
    return 0

empty_function()
start_time = timer()
#empty_function()
end_time = timer()
print(end_time - start_time)