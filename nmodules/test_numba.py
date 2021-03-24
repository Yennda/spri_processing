from numba import njit, jit, cuda
import numpy as np

@njit
def ident_np_nb(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit
def ident_np(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@jit(nopython=True, nogil = True,parallel = True)
def ident_loops_nb(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r

def ident_loops(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r

