import your_python_is_slow as ypis
import numpy as np
from time import perf_counter

arr = np.arange(2000000000, dtype=np.float64)

N = 5

for _ in range(N):
    time_start = perf_counter()
    res = np.std(arr)
    time_duration = perf_counter() - time_start
    print(f'Numpy        > took {time_duration:.3f} seconds => {res:.3f}')

for _ in range(N):
    time_start = perf_counter()
    res = ypis.fast_standard_deviation_avx2(arr)
    time_duration = perf_counter() - time_start
    print(f'SIMD AVX-2   > took {time_duration:.3f} seconds => {res:.3f}')
