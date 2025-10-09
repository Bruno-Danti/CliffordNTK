import numpy as np
from scipy.linalg.blas import dsyrk

def self_kernel(G: np.ndarray) -> np.ndarray:
    """Returns the (upper part of the) symmetric matrix K = G @ G.T"""
    return dsyrk(alpha=1.0, a=G, lower=False)

def other_kernel(G_1: np.ndarray, G_2: np.ndarray) -> np.ndarray:
    return G_1 @ G_2