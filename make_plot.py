from file_list_utils import read_multi_list
from ising_model import N_temp_step, T_max, T_min
import numpy as np
import matplotlib.pyplot as plt


def main():
    multi_list = read_multi_list('values_for_4x4.txt')
    temp_list = np.linspace(T_min, T_max, N_temp_step)
    plt.plot(temp_list, [x[0] for x in multi_list])
    plt.savefig('average_energy.png')
    plt.cla()
    plt.plot(temp_list, [x[4] for x in multi_list] )
    plt.savefig('average_abs_magnetism.png')
    plt.cla()
    plt.plot(temp_list, [x[-2] for x in multi_list])
    plt.savefig('average_heat_capacity.png')
    plt.cla()
    plt.plot(temp_list, [x[-1] for x in multi_list])
    plt.savefig('average_suspectibility_measured.png')
    plt.cla()

if __name__ == '__main__':
    main()