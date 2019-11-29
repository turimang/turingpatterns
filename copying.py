import numpy as np
import matplotlib.pyplot as plt
import os

def laplacian(Z, dx):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2



def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    ax.set_axis_off()


def turing_pattern(a=2.8e-4, b=5e-3, tau=2, k=0):
    size = 100  # size of the 2D grid
    dx = 2. / size  # space step

    T = 6.0  # total time
    dt = .001  # time step
    n = int(T / dt)  # number of iterations

    U = np.load(os.path.join('initialmatrices', 'U_initial.npy'))
    V = np.load(os.path.join('initialmatrices', 'V_initial.npy'))
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    step_plot = n // 6
    # We simulate the PDE with the finite difference
    # method.
    for i in range(n):
        # We compute the Laplacian of u and v.
        deltaU = laplacian(U, dx)
        deltaV = laplacian(V, dx)
        # We take the values of u and v inside the grid.
        Uc = U[1:-1, 1:-1]
        Vc = V[1:-1, 1:-1]
        # We update the variables.
        U[1:-1, 1:-1], V[1:-1, 1:-1] = \
            Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k),\
            Vc + dt * (b * deltaV + Uc - Vc) / tau
        # Neumann conditions: derivatives at the edges
        # are null.
        for Z in (U, V):
            Z[0, :] = Z[1, :]
            Z[-1, :] = Z[-2, :]
            Z[:, 0] = Z[:, 1]
            Z[:, -1] = Z[:, -2]

        # We plot the state of the system at
        # 9 different times.
        if i % step_plot == 0 and i < 6 * step_plot:
            ax = axes.flat[i // step_plot]
            show_patterns(U, ax=ax)
            ax.set_title(f'$t={i * dt:.2f}$')
            np.save(os.path.join('data', str(i/step_plot)), U)
    plt.show()



