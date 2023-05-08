import numpy as np
import math
from ising_utils import magnetization_grid, random_lattice_point, \
    maybe_switch_spin, grid_hamiltonian
import pickle
from driver import THROW_AWAY_COUNT, J, LATTICE_SIZE, T_max, T_min, N_temp_step, NUM_SAMPLES


def main_mcmc_loop(grid, steps, BETA):
    sum_energy = 0
    sum_energy_squared = 0
    sum_magnetism = 0
    sum_magnetism_squared = 0
    sum_magnetism_abs = 0
    num_point = math.pow(len(grid), 2)
    energy_after_maybe_flip = None
    result_arr = []
    for step_counter in range(1, steps + 1):
        for metrop in range(int(len(grid)/2)):
            random_x, random_y = random_lattice_point(grid)
            maybe_switch_spin(grid, random_x, random_y, BETA,
                              energy_after_maybe_flip, J)

        if step_counter > THROW_AWAY_COUNT:
            energy_after_maybe_flip = grid_hamiltonian(grid, J)
            magnetism_after_maybe_flip = magnetization_grid(grid)
            sum_energy += energy_after_maybe_flip
            sum_energy_squared += math.pow(energy_after_maybe_flip, 2)
            sum_magnetism += magnetism_after_maybe_flip
            sum_magnetism_abs += abs(magnetism_after_maybe_flip)
            sum_magnetism_squared += math.pow(magnetism_after_maybe_flip, 2)

            result_arr = [x / (num_point * (step_counter - THROW_AWAY_COUNT))
                          for x in
                          [sum_energy, sum_energy_squared,
                           sum_magnetism, sum_magnetism_abs,
                           sum_magnetism_squared]]

        if step_counter % 5000 == 0:
            print(result_arr, 'on step:', step_counter, '/', steps)

    return result_arr


def write_ising():
    final_observable_values = []
    num_point = math.pow(LATTICE_SIZE, 2)
    for temp in np.linspace(T_min, T_max, N_temp_step):
        grid = np.random.choice([-1, 1], (LATTICE_SIZE, LATTICE_SIZE))
        raw_results = main_mcmc_loop(grid, NUM_SAMPLES, 1 / temp)
        heat_capacity = (raw_results[1] - (
                math.pow(raw_results[0], 2) * num_point)) * math.pow(temp,
                                                                     -2)
        suspectibility = (raw_results[4] - (
                math.pow(raw_results[2], 2) * num_point)) * math.pow(temp,
                                                                     -1)
        observable_values = [
            raw_results[0],
            raw_results[3],
            heat_capacity,
            suspectibility
        ]

        final_observable_values.append(observable_values)

    with open('ising_data_dump.pkl', 'wb') as f:
        pickle.dump(final_observable_values, f)
