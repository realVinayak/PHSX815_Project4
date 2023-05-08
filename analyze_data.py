import numpy as np
from driver import T_max, T_min, N_temp_step
from analyze_utils import generate_function_to_minimize, sigmoid, \
    shifted_gaussian
import pickle
import matplotlib.pyplot as plt
from scipy import optimize

T_range = np.linspace(T_min, T_max, N_temp_step)
T_Cur = 2.27

plot_list = [
    {
        'title': 'Average Energy Per Spin Versus Temperature',
        'ylabel': 'Average Energy Per Spin',
        'ystep': None,
    },
    {
        'title': 'Average Absolute Magnetism Per Spin Versus Temperature',
        'ylabel': 'Average Absolute Magnetism Per Spin',
        'ystep': None,
    },
    {
        'title': 'Average Heat Capacity Per Spin Versus Temperature',
        'ylabel': 'Average Heat Capacity Per Spin',
        'ystep': None,
    },
    {
        'title': 'Average Magnetic Susceptibility Per Spin Versus Temperature',
        'ylabel': 'Average Magnetic Susceptibility Per Spin',
        'ystep': 0.5
    }
]


def analyze_data():
    with open('ising_data_dump.pkl', 'rb') as f:
        ising_data_dump = np.array(pickle.load(f))
    ising_data_transpose = np.transpose(ising_data_dump, (1, 0))

    analyzer(plot_list[0], ising_data_transpose[0], [-1, 2.5, 1, 1], sigmoid)
    analyzer(plot_list[1], ising_data_transpose[1], [0.8, 2.5, 1, 1], sigmoid)
    analyzer(plot_list[2], ising_data_transpose[2], [2.5, 1, 1],
             shifted_gaussian)
    analyzer(plot_list[3], ising_data_transpose[3], [2.5, 3, 3],
             shifted_gaussian)


def make_plots(spec, data):
    plt.cla()
    plt.plot(T_range, data, 'r', marker='*', linewidth=3, markersize=6)
    plt.xticks([round(x, 1) for x in np.arange(T_min, T_max + 0.5, step=0.5)])
    ystep = 0.1 if spec['ystep'] is None else spec['ystep']
    plt.yticks( [round(x, 1) for x in np.arange(min(data), max(data) + ystep, step=ystep)])
    plt.xlabel('Temperature')
    plt.ylabel(spec['ylabel'])
    plt.title(spec['title'])


def analyzer(spec, data, initial_guess, model_function):
    make_plots(spec, data)
    objective = generate_function_to_minimize(data, T_range, model_function)
    result = optimize.minimize(objective, np.array(initial_guess)).x
    test_function = model_function(*result)
    print(result)
    new_t_range = np.linspace(T_min, T_max, 50)
    plt.plot(new_t_range,
             [test_function(data_point) for data_point in new_t_range],
             color='black', linestyle='dashed')
    plt.vlines(T_Cur, min(data), max(data),
               color='blue', linewidth=1, linestyle='dotted')
    plt.legend(['Numerical Values', 'Fitted Values', 'Curie Temperature (T = '
                                                     '2.27)'])
    plt.savefig(f'./outputs/{"".join(spec["title"].split(" "))}.png')


def analyze_main():
    analyze_data()


if __name__ == '__main__':
    analyze_main()
