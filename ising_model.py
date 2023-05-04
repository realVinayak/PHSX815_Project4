import numpy as np
import math
from ising_utils import magnetization_grid, perf_wrapper
import os
from time import sleep
from file_list_utils import write_multi_list
import pickle
from time import perf_counter

LATTICE_SIZES = [8, 12]
J = 1
T_min = 0.5
T_max = 5
N_temp_step = 10
N_exp = 10


def random_lattice_point(grid):
    max_value = len(grid)
    return np.random.randint(0, max_value, 2)


def sum_hamiltonian(self, neighbours):
    mapped_spin = sum([self * neighbour_spin for neighbour_spin in neighbours])
    return -1 * J * mapped_spin


def point_hamiltonian(point_x, point_y, grid):
    neighbours = [
        grid[point_x - 1][point_y],
        grid[point_x][(point_y + 1) % len(grid)],
        grid[(point_x + 1) % len(grid)][point_y],
        grid[point_x][point_y - 1]
    ]
    return sum_hamiltonian(grid[point_x][point_y], neighbours)


def grid_hamiltonian(grid):
    total_hamiltonian = 0
    for row_index in range(len(grid)):
        for col_index in range(len(grid[row_index])):
            total_hamiltonian += point_hamiltonian(row_index, col_index,
                                                   grid)
    return total_hamiltonian


def heat_capacity(grid, temp):
    return (grid_hamiltonian(grid, mode='square') - math.pow(
        grid_hamiltonian(grid), 2)) / math.pow(temp, 2)


def suspectibility(grid, temp):
    return (magnetization_grid(grid, 'square') - math.pow(
        magnetization_grid(grid, 'normal'), 2)) / temp


def maybe_switch_spin(grid, x, y, BETA):
    point_energy = point_hamiltonian(x, y, grid)
    energy_change = -2 * point_energy
    switch_prob = math.exp(-1 * energy_change * BETA)
    if energy_change <= 0 or np.random.random() < switch_prob:
        grid[x][y] *= -1
        return True
    return False


def print_grid(grid, val):
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
    sleep(0.2)


def print_and_return(val1, val2):
    # print('diff: ', abs(val2 - val1))
    return val1

def main_mcmc_prev_loop(grid, steps, BETA):
    num_point = math.pow(len(grid), 2)
    sum_energy = 0
    prev_energy = grid_hamiltonian(grid)/2
    for step_counter in range(1, steps + 1):
        for metrop in range(32):
            random_x, random_y = random_lattice_point(grid)
            if maybe_switch_spin(grid, random_x, random_y, BETA):
                prev_energy += 2*point_hamiltonian(random_x, random_y, grid)

        sum_energy += prev_energy

        if step_counter % 500 == 0:
            print('average energy: ', sum_energy / (num_point*step_counter))

    return [sum_energy/num_point]


def main_mcmc_loop(grid, steps, BETA):

    energy = grid_hamiltonian(grid)/2
    magnetism = magnetization_grid(grid)
    magnetism_absolute = abs(magnetization_grid(grid))

    sum_energy = 0

    sum_magnetism = 0
    sum_magnetism_squared = 0
    sum_magnetism_abs = 0

    num_point = math.pow(len(grid), 2)

    for step_counter in range(1, steps + 1):
        t1 = perf_counter()
        for metrop in range(int(len(grid)/2)):
            random_x, random_y = random_lattice_point(grid)
            if maybe_switch_spin(grid, random_x, random_y, BETA):
                energy += 2*point_hamiltonian(random_x, random_y, grid)
                magnetism += 2*grid[random_x][random_y]
                magnetism_absolute += abs(grid[random_x][random_y])
        sum_energy += energy
        sum_magnetism += magnetism
        sum_magnetism_squared += math.pow(magnetism, 2)
        sum_magnetism_abs += abs(magnetism)
        t2 = perf_counter()
        #print_grid(grid, '')
        #if (step_counter % 200 == 0):
        #    print('on step: ', step_counter, 'took: ', t2-t1)
    return [x / (num_point * steps) for x in
            [sum_energy, sum_magnetism,
             sum_magnetism_squared, sum_magnetism_abs]]


def main():
    final_data_list = []
    #grid = np.random.choice([-1, 1], (4, 4))
    #(main_mcmc_prev_loop(grid, 10000, 1 / 0.5))


    for _ in range(N_exp):
        per_lattice_size = []
        for LATTICE_SIZE in LATTICE_SIZES:
            observable_values = []
            for temp in np.linspace(T_min, T_max, N_temp_step):
                grid = np.random.choice([-1, 1], (LATTICE_SIZE, LATTICE_SIZE))
                observable_values.append(main_mcmc_loop(grid, 10000, 1 / temp))
                print('temp: ', temp, 'lattice_size: ', LATTICE_SIZE,
                      'experiment: ', _, 'average: ', observable_values[-1][0], 'ab mag: ', observable_values[-1][-1])
            per_lattice_size.append(observable_values)
        final_data_list.append(per_lattice_size)

    with open('ising_data_dump.pkl', 'wb') as f:
        pickle.dump(final_data_list, f)


if __name__ == '__main__':
    main()
