from test_numba import *
import timeit

# nb = timeit.timeit('ident_np_nb(15)', setup = 'from test_numba import ident_np_nb', number=1000)
# py = timeit.timeit('ident_np(15)', setup = 'from test_numba import ident_np', number=1000)


# nb = timeit.timeit('ident_loops_nb(np.array([0.2]*10000))', setup = 'from test_numba import ident_loops_nb; import numpy as np', number=100)
# py = timeit.timeit('ident_loops(np.array([0.2]*10000))', setup = 'from test_numba import ident_loops; import numpy as np', number=100)


# nb = timeit.timeit('ident_np_nb(15)', setup = 'from test_numba import ident_np_nb', number=1000)
# py = timeit.timeit('ident_np(15)', setup = 'from test_numba import ident_np', number=1000)

py = timeit.timeit('raw_diff(m, 10)', setup = 'from tools_numba import raw_diff; import numpy as np; m = np.random.random(size = [30, 30, 500])', number=10000)
nb = timeit.timeit('raw_diff_nb(m, 10)', setup = 'from tools_numba import raw_diff_nb; import numpy as np; m = np.random.random(size = [30, 30, 500])', number=10000)


print(py)
print(nb)