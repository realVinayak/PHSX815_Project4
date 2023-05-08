# PHSX815_Project4

The goal of this project is to study phase transitions in
materials using the Ising Model. The physical properties studied
in this project are *average energy per spin*, *average absolute magnetism per spin*,
*average heat capacity per spin* and *average magnetic susceptibility per spin*. Monte Carlo
Markov Chain method is used to numerically estimate these physical values for 
a lattice with lattice size = `LATTICE_SIZE` over a range of temperature. Then, 
these values are analyzed for sudden changes near the Curie Temperature which marks the
phase transition.


### Structure and Contents
- `ising_utlis.py`: Provides the core utility functions for simulating the Ising Model. Composed of following important 
functions:
  1. `point_hamiltonian(point_x, point_y, grid)`: Takes in a point `(point_x, point_y)` on the grid (lattice)
  and returns the energy of that point by calculating spin interactions with immediate neighbours.
  2. `grid_hamiltonian(grid, J)`: Takes in the grid and returns the energy of the entire grid by calling `point_hamiltonian(point_x, point_y, grid)` 
  for each point on the grid and summing the result. Also divides the result by 2 to not double count spins.
  3. `maybe_switch_spin(grid, x, y, BETA, current_energy, J)`: Takes in the grid, and a point `(x, y)`. Tries performing a spin flip at the input point
      and returns `True` if spin flip is performed otherwise returns `False`. If the spin flip decreases the energy of the grid, then the flip is performed.
      Otherwise, the spin flip is performed with probability `math.exp(-1 * energy_change * BETA)`
  4. `random_lattice_point(grid)`: Returns a random lattice point (x, y) on the grid.
  5. `print_grid(grid)`: Prints the grid for observing patterns.

- `ising_model.py`: Provides two important functions:
  1. `main_mcmc_loop(grid, steps, BETA)`:
     Runs the Ising Model simulation for a fixed
     temperature. Also keeps track of the observables 
     (physical properties that need to reported). Disregards `THROW_AWAY_COUNT`
     samples to wait for the samples to converge to the probability distribution.
  2. `write_ising()`: Performs the `main_mcmc_loop` multiple times with varying temperatures
  and computes heat capacity and susceptibility after each temperature loop is finished.
  Dumps the observable values (*average energy per spin*, *average absolute magnetism per spin*,
*average heat capacity per spin* and *average magnetic susceptibility per spin*) over multiple temperatures 
  into `ising_data_dump.pkl`
- `analyze_utils.py`: Provides important utility functions and model definitions for analyzing data.
  1. `wrap_mean_squared(data, sample_points, func)`: Applies function `func` at each sample in `sample_points` 
  and computes the mean squared error with true data `data`.
  2. `sigmoid(yoffset, xoffset, yscale, xscale)`: Returns a modified sigmoid function which is shifted to `(xoffset, yoffset)`
  and scaled vertically and horizontly by `yscale` and `xscale` respectively.
  3. `shifted_gaussian(xoffset, yscale, std_dev)`: Returns a modified Gaussian function which is shifted horizontly by `xoffset`, scaled
  vertically by `yscale` and scaled horizontly by `std_dev`.
  4. `generate_function_to_minimize(data, sample_points, model_function)`: Returns a function which takes in the parameters of the model function `model_function`
  and returns mean squared error of the parameterized model function on `data` and `sample_points`.
- `analyze_data.py`: Reads the Ising Model data output from `ising_model.py` and performs analysis and makes plots of the data. The two
     important functions in this file are:
  1. `analyzer(spec, data, initial_guess, model_function)`: Uses scipy's minimize function to find the best-fit parameters of `model_function`
  which minimize the squared error of an observable. Also plots the fitted function, the raw data and Curie Temperature. The plot for each observable is saved
  in `./outputs/`
- `driver.py`: Provides a driver wrapper to perform the following tasks:
  - Defines the following hyperparameters of Ising Model. 
    1. The lattice size - `LATTICE_SIZE x LATTICE_SIZE`. Default `LATTICE_SIZE = 4`.
    2. The spin interaction parameter - `J`. Default value of `J = 1`.
    3. The minimum temperature - `T_min`. Default value of `T_min = 0.1`.
    4. The maximum temperature - `T_max`. Default value of `T_max = 6`.
    5. The number of temperature samples between `T_min` and `T_max` - `N_temp_step`. Default value of `N_temp_step = 5`.
    6. The number of initial samples to throwaway - `THROW_AWAY_COUNT`. Default value of `THROW_AWAY_COUNT` = 7000.
    7. The number of samples to consider - `NUM_SAMPLES`. Default value of `NUM_SAMPLES = 20000`
  - Generates the set of temperature as `np.linspace(T_min, T_max, N_temp_step)`, and calls `write_ising()` from `ising_model_main.py` which runs Ising Model simulation for each value of
           temperature in the set and saves *average energy per spin*, *average absolute magnetism per spin*,
*average heat capacity per spin* and *average magnetic susceptibility per spin* for each temperature in `ising_data_dump.pkl`.
    - Runs `analyze_main()` from `analyze_data.py` which generates separate plots for each average observables over temperature. Also
    performs best fit function analysis and saves the resulting plots in `./outputs/`

### Execution
To run, type `python3 driver.py`. Currently, the lattice size used in code is `4 x 4`. However, the paper uses
`16 x 16` as the lattice size. This is done because the execution times is longer for `16 x 16`.