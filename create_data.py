import os
import numpy as np
from plot_data import plot

lst = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def add_noise(i):
    data = np.load(os.path.join('data', str(i) + '.npy'))
    data += np.random.normal(0, 0.05, (100, 100))
    return data

def main(lst):
    for i in lst:
        noisy_data = add_noise(i)
        np.save(os.path.join('noisy_data', str(i)), noisy_data)

main(lst)