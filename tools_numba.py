import numpy as np
from numba import jit, cuda
import scipy
import scipy.signal
import time


# @jit(nopython=True, parallel=True)
@jit(nopython=True)
def raw_diff(data_raw, k, idea3d):
    # data_diff_std = []
    raw_diff = np.zeros_like(data_raw)
    data_corr = np.zeros_like(data_raw)

    difference = np.zeros_like(data_raw[:, :, 0])
    current = np.zeros_like(difference)
    previous = np.zeros_like(difference)

    for f in range(data_raw.shape[2]):
        current = np.sum(
            data_raw[:, :, f - k + 1: f + 1],
            axis=2
        ) / k
        previous = np.sum(
            data_raw[:, :, f - 2 * k + 1: f - k + 1],
            axis=2
        ) / k
        # difference = np.add(current, - previous)
        raw_diff[:, :, f] = np.add(current, - previous)
        # data_diff_std.append(np.std(difference))

    return data_corr

# @jit(nopython=True)
@jit(nopython=True, parallel = True)
def frame_diff(data_raw, k, f):
    current = np.sum(
        data_raw[:, :, f - k + 1: f + 1],
        axis=2
    ) / k
    previous = np.sum(
        data_raw[:, :, f - 2 * k + 1: f - k + 1],
        axis=2
    ) / k
    difference = current - previous
    return difference, np.std(difference)
