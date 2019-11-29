import numpy as np
import os
from plot_data import plot

def calc_error(target, noise):
    diff = 0
    size = len(target)
    diff_matrix = np.zeros((size, size))
    for i in range(len(target)):
        for j in range(len(target[i])):
            error = (target[i][j] - noise[i][j])
            diff += error**2
            diff_matrix[i][j] = error
    return diff, diff_matrix


i = 1.0
data = np.load(os.path.join('data', str(i)+'.npy'))
noisy_data = np.load(os.path.join('noisy_data', str(i)+'.npy'))
error_total, error_matrix = calc_error(data, noisy_data)

plot(data, i)
plot(noisy_data, i)
plot(error_matrix, i)