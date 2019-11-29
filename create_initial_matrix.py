import numpy as np

size = 100
U = np.random.rand(size, size)
V = np.random.rand(size, size)

np.save('U_initial', U)
np.save('V_initial', V)
