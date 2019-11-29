import numpy as np
import os

def laplacian(Z, dx):
    """
    Calculates laplace operator for particular matrix
    :param Z: Combination of two initial matrices
    :param dx: float space step
    :return: Boundaries of matrix
    """
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2


def turing(a, b, time):
    """
    calculates the turing patterns over time
    :param a: concentration of a
    :param b: concentration of b
    :param time: time that we want result for
    :return: turing pattern for particular time
    """
    size = 100
    dx = 2. / size  # space step
    tau = 2.00
    k = 0

    T = 6.0  # total time
    dt = .001  # time step
    n = int(T / dt)  # number of iterations
    step_plot = n // 6

    U = np.load(os.path.join('initialmatrices', 'U_initial.npy'))
    V = np.load(os.path.join('initialmatrices', 'V_initial.npy'))

    for i in range(n):

        # We compute the Laplacian of u and v.
        deltaU = laplacian(U, dx)
        deltaV = laplacian(V, dx)
        # We take the values of u and v inside the grid.
        Uc = U[1:-1, 1:-1]
        Vc = V[1:-1, 1:-1]
        # We update the variables.
        U[1:-1, 1:-1], V[1:-1, 1:-1] = \
            Uc + dt * (a * deltaU + Uc - Uc ** 3 - Vc + k), \
            Vc + dt * (b * deltaV + Uc - Vc) / tau

        # Neumann conditions: derivatives at the edges
        # are null.
        for Z in (U, V):
            Z[0, :] = Z[1, :]
            Z[-1, :] = Z[-2, :]
            Z[:, 0] = Z[:, 1]
            Z[:, -1] = Z[:, -2]


        if i * dt == time:
            return U

