import numpy as np

def fn(x):
    y = 0
    l = len(x)
    for i in range(l - 1):
        x[i] = x[i] - x[i+1]
    return x