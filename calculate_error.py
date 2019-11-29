import numpy as np


def calc_error(target, noise):
    """
    :param target: 2D matrix, data produced using potential a and b parameters
    :param noise: 2D matrix, data for comparison
    :return: sum of squares of error distance between target and noise at each coordinate
    """

    diff = 0
    size = len(target)
    diff_matrix = np.zeros((size, size))
    for i in range(len(target)):
        for j in range(len(target[i])):
            error = (target[i][j] - noise[i][j])
            diff += error**2
            diff_matrix[i][j] = error
    return diff
