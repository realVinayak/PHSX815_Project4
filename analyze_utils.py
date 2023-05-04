import math


# Returns log likelihood ratio given callbacks for first probability and
# second probability and measurement
def get_log_likelihood_ratio(measurement, get_first_prob, get_second_prob):
    first_prob = get_first_prob(measurement)
    second_prob = get_second_prob(measurement)
    log_likelihood_ratio = math.log(
        max(first_prob, 10 ** (-50)) / max(second_prob, 10 ** (-50)))
    return log_likelihood_ratio


# Returns false negative rate given llr measurements and lambda threshold
def get_fnr(llr_measurements, lambda_threshold):
    llr_passed_count = 0
    for llr in llr_measurements:
        if llr > lambda_threshold:
            break
        llr_passed_count += 1
    return llr_passed_count / len(llr_measurements)


# Takes in the histogram in the form of hist_data and hist_bins. Given a
# sample, we find the corresponding bin in the histogram's bins (hist_bins).
# Then, the height for that particular bin is read and returned as the
# probability. Assumes that hist_data is already normalized.
def get_probability_from_hist(hist_data, hist_bins, sample):
    bin_index = 0
    probability = 0
    if hist_bins[0] <= sample <= hist_bins[-1]:
        while bin_index <= len(hist_bins) - 2:
            if hist_bins[bin_index] <= sample <= hist_bins[bin_index + 1]:
                break
            bin_index += 1
        probability = hist_data[bin_index]
    return probability

import numpy as np


# Uses numpy's histogram generator and returns
# data that can be read efficiently by matplotlib
# Reason: matplotlib's histogram is slow. numpy's is fast.
def get_histogram_data(measurements, samples):
    temp_hist = list(np.histogram(measurements, samples, density=True))
    temp_hist_probs = list(temp_hist[0])
    temp_bin_flatten = []
    past_bin = temp_hist[1][0]
    for _bin in temp_hist[1][1:]:
        temp_bin_flatten.append((_bin + past_bin) / 2)
        past_bin = _bin
    return temp_hist_probs, temp_bin_flatten