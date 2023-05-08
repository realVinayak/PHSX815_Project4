import numpy as np
import math
from time import perf_counter, sleep
import os

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

def random_lattice_point(grid):
    max_value = len(grid)
    return np.random.randint(0, max_value, 2)


def hamiltonian(self, neighbours, J):
    mapped_spin = sum([self * neighbour_spin for neighbour_spin in neighbours])
    return -1 * J * mapped_spin


def grid_hamiltonian(grid, J):
    total_hamiltonian = 0
    for row_index in range(len(grid)):
        for col_index in range(len(grid[row_index])):
            neighbours = [
                grid[row_index - 1][col_index],
                grid[row_index][(col_index + 1) % len(grid)],
                grid[(row_index + 1) % len(grid)][col_index],
                grid[row_index][col_index - 1]
            ]
            total_hamiltonian += hamiltonian(grid[row_index][col_index],
                                             neighbours, J)
    return total_hamiltonian / 2


def maybe_switch_spin(grid, x, y, BETA, current_energy, J):
    if current_energy is None:
        current_energy = grid_hamiltonian(grid, J)
    grid[x][y] *= -1
    later_energy = grid_hamiltonian(grid, J)
    energy_change = later_energy - current_energy
    switch_prob = math.exp(-1 * energy_change * BETA)
    if energy_change <= 0 or np.random.random() < switch_prob:
        return True
    grid[x][y] *= -1
    return False


def print_grid(grid):
    white_block = '\u25A1'
    black_block = '\u25A0'
    os.system('clear')
    for row in grid:
        for col in row:
            if col == 1:
                print(white_block, end='')
            else:
                print(black_block, end='')
        print('\n', end='')
    sleep(0.5)