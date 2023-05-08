import numpy as np
import math


def wrap_mean_squared(data, sample_points, func):
    predicted_values = np.array([func(sample) for sample in sample_points])
    diff = data - predicted_values
    error = math.pow(np.linalg.norm(diff), 2)
    return error


def sigmoid(yoffset, xoffset, yscale, xscale):
    return lambda x: yoffset + (
                yscale / (1 + math.exp(-1 * xscale * (x - xoffset))))


def shifted_gaussian(xoffset, yscale, std_dev):
    return lambda x: (yscale/abs(std_dev))*(math.exp(-1*0.5*(math.pow((x-xoffset)/(std_dev), 2))))


def generate_function_to_minimize(data, sample_points, model_function):
    return lambda params: wrap_mean_squared(data, sample_points,
                                            model_function(*params))
