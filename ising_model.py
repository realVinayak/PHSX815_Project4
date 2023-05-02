import numpy as np
import math
from ising_utils import magnetization_grid, perf_wrapper
import os
from time import sleep
from file_list_utils import write_multi_list

LATTICE_SIZE = 8
J = 1
T_min = 0.5
T_max = 5
N_temp_step = 10


class AverageObservable:
    def __init__(self, name, update):
        self.name = name
        self.update = update
        self.value = 0

    def __str__(self):
        return f'{self.name}: {self.value}'


def random_lattice_point(grid):
    max_value = len(grid)
    return np.random.randint(0, max_value, 2)


def hamiltonian(self, neighbours):
    mapped_spin = sum([self * neighbour_spin for neighbour_spin in neighbours])
    return -1 * J * mapped_spin


def grid_hamiltonian(grid, mode='normal'):
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
                                             neighbours)
    return total_hamiltonian / 2 if mode == 'normal' else math.pow(
        total_hamiltonian / 2, 2)


def heat_capacity(grid, temp):
    return (grid_hamiltonian(grid, mode='square') - math.pow(
        grid_hamiltonian(grid), 2)) / math.pow(temp, 2)


def suspectibility(grid, temp):
    return (magnetization_grid(grid, 'square') - math.pow(
        magnetization_grid(grid, 'normal'), 2)) / temp


def maybe_switch_spin(grid, x, y, BETA):
    current_energy = grid_hamiltonian(grid)
    grid[x][y] *= -1
    later_energy = grid_hamiltonian(grid)
    energy_change = later_energy - current_energy
    switch_prob = math.exp(-1 * energy_change * BETA)
    if energy_change <= 0 or np.random.random() < switch_prob:
        return True
    grid[x][y] *= -1
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
    sleep(0.5)


def main_mcmc_loop(grid, steps, BETA):
    average_energy = 0
    average_energy_squared = 0
    average_magnetism = 0
    average_magnetism_squared = 0
    average_magnetism_abs = 0
    heat_capacity_measured = 0
    suspectibility_measured = 0
    num_point = math.pow(len(grid), 2)

    for step_counter in range(1, steps + 1):
        for metrop in range(32):
            random_x, random_y = random_lattice_point(grid)
            maybe_switch_spin(grid, random_x, random_y, BETA)

        average_energy = ((average_energy * (
                    step_counter - 1) + grid_hamiltonian(grid)) / step_counter)
        average_energy_squared = ((average_energy_squared * (
                    step_counter - 1) + grid_hamiltonian(grid,
                                                         mode='square')) / step_counter)
        average_magnetism = ((average_magnetism * (
                    step_counter - 1) + magnetization_grid(grid,
                                                           mode='normal')) / step_counter)
        average_magnetism_squared = ((average_magnetism_squared * (
                    step_counter - 1) + magnetization_grid(grid,
                                                           mode='square')) / step_counter)
        average_magnetism_abs = ((average_magnetism_abs * (
                    step_counter - 1) + magnetization_grid(grid,
                                                           mode='abs')) / step_counter)
        heat_capacity_measured = ((average_energy_squared - (
                    average_energy * average_energy)) * (math.pow(BETA, 2)))
        suspectibility_measured = (average_magnetism_squared - (
                    average_magnetism * average_magnetism)) * (
                                      math.pow(BETA, 1))

        if step_counter % 500 == 0:
            print('average energy: ', average_energy / num_point)
            print('average absolute magnetism: ',
                  average_magnetism_abs / num_point)
            print('heat capacity measured: ',
                  heat_capacity_measured / num_point)
            print('temperature: ', 1 / BETA)
            print(step_counter, '/', steps)

    return [x / num_point for x in
            [average_energy, average_energy_squared, average_magnetism,
             average_magnetism_squared, average_magnetism_abs,
             heat_capacity_measured,
             suspectibility_measured]]


def main():
    observable_values = []
    for temp in np.linspace(T_min, T_max, N_temp_step):
        grid = np.random.choice([-1, 1], (LATTICE_SIZE, LATTICE_SIZE))
        observable_values.append(main_mcmc_loop(grid, 10000, 1 / temp))
    write_multi_list(observable_values, 'values_for_4x4.txt')


if __name__ == '__main__':
    main()
