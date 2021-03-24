import numpy as np
import numba
from numba import jit, cuda
import time


# @jit(nopython=True, parallel=True)
@jit(nopython=True)
def raw_diff_nb(data_raw, k):
    # data_diff_std = []
    raw_diff =  np.zeros_like(data_raw[1, 1, :])
    difference =  np.zeros_like(data_raw[:, :, 0])
    current = np.zeros_like(difference)
    previous = np.zeros_like(difference)


    for f in range(data_raw.shape[2]):
        current = np.sum(
            data_raw[:, :, f - k + 1: f + 1],
            axis=2
        )
        previous = np.sum(
            data_raw[:, :, f - 2 * k + 1: f - k + 1],
            axis=2
        )

        # difference = np.add(current, - previous)
        raw_diff[f] = np.add(current, -previous)
        # data_diff_std.append(np.std(difference))

    return raw_diff
# @jit(nopython=True)
# def raw_diff_nb(data_raw, k):
#
#     # data_diff_std = []
#     raw_diff = np.zeros_like(data_raw[1, 1, :])
#     # difference =  np.zeros_like(data_raw[:, :, 0])
#     # current = np.zeros_like(difference)
#     # previous = np.zeros_like(difference)
#     current = 0.0
#     previous = 0.0
#
#     for f in range(data_raw.shape[2]-1):
#         # current = np.sum(
#         #     data_raw[:, :, f - k + 1: f + 1],
#         #     axis=2
#         # )
#         # previous = np.sum(
#         #     data_raw[:, :, f - 2 * k + 1: f - k + 1],
#         #     axis=2
#         # )
#         current = np.sum(
#             data_raw[:, :, f])
#         previous = np.sum(
#             data_raw[:, :, f + 1])
#         # difference = np.add(current, - previous)
#         raw_diff[f] = np.add(current, -previous)
#         # data_diff_std.append(np.std(difference))
#
#     return raw_diff

def raw_diff(data_raw, k):

    # data_diff_std = []
    raw_diff = np.zeros_like(data_raw[1, 1, :])
    # difference =  np.zeros_like(data_raw[:, :, 0])
    # current = np.zeros_like(difference)
    # previous = np.zeros_like(difference)
    current = 0.0
    previous = 0.0

    for f in range(data_raw.shape[2]-1):
        # current = np.sum(
        #     data_raw[:, :, f - k + 1: f + 1],
        #     axis=2
        # )
        # previous = np.sum(
        #     data_raw[:, :, f - 2 * k + 1: f - k + 1],
        #     axis=2
        # )
        current = np.sum(
            data_raw[:, :, f])
        previous = np.sum(
            data_raw[:, :, f + 1])
        # difference = np.add(current, - previous)
        raw_diff[f] = np.add(current, -previous)
        # data_diff_std.append(np.std(difference))

    return raw_diff


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
