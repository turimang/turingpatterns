import numpy as np
import os

def create_initial_matrix(size):
    U = np.random.rand(size, size)
    V = np.random.rand(size, size)

    np.save(os.path.join('initialmatrices', 'U_initial'), U)
    np.save(os.path.join('initialmatrices', 'V_initial'), V)


create_initial_matrix(100)