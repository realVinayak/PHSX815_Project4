import numpy as np
import ast
from ising_model import LATTICE_SIZES, T_min, T_max, N_temp_step
from file_list_utils import read_multi_list
from analyze_utils import get_log_likelihood_ratio, get_histogram_data, get_fnr, \
    get_probability_from_hist
import matplotlib.pyplot as plt
import pickle

ALPHA = 0.90
MIN_X = -float('inf')
MAX_X = float('inf')

T_index = 0
ABS_MAG = 1
lattice_1_index = 0
lattice_2_index = 1

'''
Assumptions:
Assumes data is of type [lattice_size][temperature][observable][experiments]
'''


def main():
    analyze_atomic()


# Plots of histogram of LLR for hypothesis 1 and hypothesis 2.
# Also stores the final plot in ./outputs/histograms
def plot_atomic_result(llr_dist_1, llr_dist_2, lambda_alpha):
    file_name = f'hist_dump.png'
    llr_dist_1_probs, llr_dist_1_bins = get_histogram_data(llr_dist_1,
                                                           100)
    llr_dist_2_probs, llr_dist_2_bins = get_histogram_data(llr_dist_2,
                                                           100)
    plt.plot([], 'r')
    plt.plot([], 'g')
    plt.plot([], 'black')
    plt.plot(llr_dist_1_bins, llr_dist_1_probs, 'r', linewidth=0.8)
    plt.vlines(lambda_alpha, 0,
               max(max(llr_dist_1_probs), max(llr_dist_2_probs)),
               color='black', linewidth=1)
    plt.plot(llr_dist_2_bins, llr_dist_2_probs, 'g', linewidth=0.8
             )
    plt.xlabel('\u03BB = L(H2)/L(H1)')
    plt.ylabel('Probability')
    plt.legend(['P(\u03BB|H1)', 'P(\u03BB|H2)',
                f'\u03BB\u1D45 = {round(lambda_alpha, 2)}'])
    plt.savefig(file_name)
    plt.show()
    plt.clf()


# Takes in num_meas and depth. Loads samples from the corresponding file
# and calculates and returns the false negative rate.
def analyze_atomic():
    with open('ising_data_dump.pkl', 'rb') as f:
        read_data_t = pickle.load(f)

    read_data = np.transpose(np.array(read_data_t), (1, 2, 3, 0))
    print(np.array(read_data_t).shape)

    hypothesis_1 = read_data[lattice_1_index][T_index][ABS_MAG]
    hypothesis_2 = read_data[lattice_2_index][T_index][ABS_MAG]

    print(read_data[lattice_1_index][T_index][1])
    print(read_data[lattice_2_index][T_index][1])
    print([sum(read_data[lattice_1_index][T_index][abs_]) - sum(read_data[lattice_2_index][T_index][abs_]) for abs_ in range(0, 4)])


    '''
    hypothesis_1_data, hypothesis_1_bins = np.histogram(hypothesis_1, 100,
                                                        density=True)
    hypothesis_2_data, hypothesis_2_bins = np.histogram(hypothesis_2, 100,
                                                        density=True)
    llr_distribution_h1 = [get_log_likelihood_ratio(raw_meas,
                                                    lambda
                                                        x: get_probability_from_hist(
                                                        hypothesis_2_data,
                                                        hypothesis_2_bins, x),
                                                    lambda
                                                        x: get_probability_from_hist(
                                                        hypothesis_1_data,
                                                        hypothesis_1_bins, x))
                           for raw_meas in hypothesis_1]
    llr_distribution_h2 = [get_log_likelihood_ratio(raw_meas,
                                                    lambda
                                                        x: get_probability_from_hist(
                                                        hypothesis_2_data,
                                                        hypothesis_2_bins, x),
                                                    lambda
                                                        x: get_probability_from_hist(
                                                        hypothesis_1_data,
                                                        hypothesis_1_bins, x))
                           for raw_meas in hypothesis_2]

    llr_distribution_h1 = list(
        filter(lambda llr: MIN_X <= llr <= MAX_X, llr_distribution_h1))
    llr_distribution_h2 = list(
        filter(lambda llr: MIN_X <= llr <= MAX_X, llr_distribution_h2))
    llr_distribution_h1.sort()
    llr_distribution_h2.sort()
    _lambda_alpha = np.percentile(llr_distribution_h1, 100 * ALPHA)
    fnr = get_fnr(llr_distribution_h2, _lambda_alpha)

    print('lamda alpha: ', _lambda_alpha)
    print('false negative rate: ', fnr)

    temp_list = np.linspace(T_min, T_max, N_temp_step)
    print(read_data[lattice_2_index].shape)
    plt.plot(temp_list, np.transpose(read_data[lattice_2_index], (2, 1, 0))[0][-1])
    plt.savefig('random1.png')
    plt.cla()
    plt.plot(temp_list,
             np.transpose(read_data[lattice_1_index], (2, 1, 0))[0][-1])
    plt.savefig('random2.png')

    #plot_atomic_result(llr_distribution_h1, llr_distribution_h2, _lambda_alpha)
    return round(fnr, 2)
'''
if __name__ == '__main__':
    main()