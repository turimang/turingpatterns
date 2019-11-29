import unittest
import numpy as np
import os
from optimising import optimise


def test_initial_matrix():
    matrix = np.load(os.path.join('initialmatrices', 'U_initial.npy'))
    assert len(matrix) == 100

def test_turing_pattern():
    pattern = np.load(os.path.join('data', '1.0.npy'))
    matrix = np.load(os.path.join('initialmatrices', 'U_initial.npy'))
    assert pattern[0][0] != matrix[0][0] or pattern [0][1] != matrix[0][1]

def test_make_noise():
    pattern = np.load(os.path.join('data', '1.0.npy'))
    noisy = np.load(os.path.join('noisy_data', '1.0.npy'))
    equal = 0
    for i in range(10):
        if pattern[i][i] == noisy[i][i]:
            equal += 1
    assert equal < 2

def test_optimisation():
    a_est, b_est = optimise(time=1.0, a_initial=0.0, a_final=1e-3, b_initial=0.0, b_final=1e-2,
                            iters=5)
    assert 1e-4 < a_est < 1e-3 and 1e-3 < b_est < 1e-2

if __name__ == '__main__':
    unittest.main()
