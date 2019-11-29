from turing_anna import turing
from plot_data import plot
from calculate_error import calc_error
import numpy as np
import os

# target parameters
# a = 2.8e-4
# b = 5e-3


def optimise(time = 1.0, a_initial = 0.0, a_final = 1e-3, b_initial = 0.0, b_final = 1e-2, iters=10):
    a_step = (a_final-a_initial) / iters
    b_step = (b_final-b_initial) / iters

    a = a_initial
    b = b_initial

    data = np.load(os.path.join('noisy_data', str(time) + '.npy'))

    error = 100000
    result = None
    a_final = None
    b_final = None
    for i in range(iters):
        b = b_initial + i * b_step # incrementing b
        for j in range(iters):
            a = a_initial + j * a_step # incrementing a from minimum to maximum for each b value
            result = turing(a, b, time) # getting the turing pattern for that particular time
            error_new = calc_error(result, data) # calculating the error between the data we have and the turing pattern
                                                 # of the given parameters
            print(a, b, error_new)
            if error_new < error: # saving the parameters if the error is lower
                error = error_new
                final_result = result
                a_final = a
                b_final = b

    plot(data, time) # plotting the target data
    plot(final_result, time) # plotting the data that would be produced from the target data for comparison
    print(a_final, b_final, error)
    return a_final, b_final
