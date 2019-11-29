import numpy as np
import os

size = 100
U = np.random.rand(size, size)
V = np.random.rand(size, size)

np.save(os.path.join('initialmatrices', 'U_initial'), U)
np.save(os.path.join('initialmatrices', 'V_initial'), V)
