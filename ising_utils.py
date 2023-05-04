import numpy as np
import math
from time import perf_counter


def magnetization_grid(grid):
    num_up = sum([x + 1 for x in np.ndarray.flatten(grid)]) / 2
    num_down = math.pow(len(grid), 2) - num_up
    magnetization = num_up - num_down
    return magnetization


# wrapper to estimate performance
def perf_wrapper(func, message):
    t_init = perf_counter()
    callback_return = func()
    t_final = perf_counter()
    print(t_final - t_init, message)
    return callback_return
