# import time
# import numpy
#
# cimport numpy
#
# ctypedef numpy.int_t DTYPE_t
#
#
# def do_calc(numpy.ndarray[DTYPE_t, ndim = 1] arr):
#     cdef int maxval
#     cdef unsigned long long int total
#     cdef int k
#     cdef double t1, t2, t
#     cdef int arr_shape = arr.shape[0]
#
#     t1 = time.time()
#
#     #    for k in arr:
#     #        total = total + k
#
#     for k in range(arr_shape):
#         total = total + arr[k]
#         print("Total =", total)
#         t2 = time.time()
#         t = t2 - t1
#         print("%.20f" % t)

import numpy
cimport numpy


def fn(numpy.ndarray[numpy.float_t, ndim = 1] x):
    cdef int l = len(x)
    cdef float y = 0

    for i in range(l - 1):
        x[i] = x[i] - x[i+1]
    return y