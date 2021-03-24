from cmodules.first_cython import fn

import numpy as np

from cmodules.first_py import fn as fnpy
import timeit

# x = np.random.random(10)
# x = np.array([1, 2.3, 1.8, 5, 17, 3.4, 0.5])
# print(x)
# print(fnpy(x))
# print(fn(x))

cy = timeit.timeit('fn(np.array([1, 2.3, 1.8, 5, 17, 3.4, 0.5]))', setup='from cmodules.first_cython import fn; import numpy as np', number=int(1e6))
py = timeit.timeit('fn(np.array([1, 2.3, 1.8, 5, 17, 3.4, 0.5]))', setup='from cmodules.first_py import fn; import numpy as np', number=int(1e6))

print(cy,py)
print(py/cy)


