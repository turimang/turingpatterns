import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def construct_laplace_matrix_1d(N, h):
    """
    Function generates an NxN 1D laplace matrix
    :param N: Number of iterations
    :param h: Spacial step
    :return: N x N 1D laplace matrix
    """
    e = np.ones(N)
    diagonals = [e, -2*e, e]
    offsets = [-1, 0, 1]
    L = scipy.sparse.spdiags(diagonals, offsets, N, N) / h**2
    return L


eps = 1
h = 0.05
alpha = 0.5
k = 0.4 * h ** 2 / eps
Tf = 42.0
x = np.arange(0 + h, 20 - h, h)


def solve_turing():
    eps = 1.0
    h = 0.05
    alpha: int = 3
    k = 0.4 * h**2 / eps
    Tf = 42.0
    x = np.arange(0+h, 20-h, h)

    N = len(x)
    L = construct_laplace_matrix_1d(N, h)
    bc = np.concatenate(([1], np.zeros(N-1)))/h**2
    out = []
    u = (x<3).astype('float64')

    fig, ax = plt.subplots()
    ln, = plt.plot(x, u)
    ax.set_ylim(0, 1.1)

    def getsols():
        for i in range(int(Tf/k)):
            u_new = u + k*(eps*(L*u + bc) + (u-u**alpha))
            out.append([u_new])
            u[:] = u_new
        return u
    u = getsols()
    out.append([u])
    return u, x, out


u, x, out = solve_turing()
plt.figure()
plt.plot(x, out[0][0])
plt.plot(x, out[int(Tf/(4.0*k))][0])
plt.plot(x, out[-1][0])
plt.show()







# eps = 1.0
# h = 0.05
# start = 0+h
# end = 20-h
# N = (end - start)/h
# alpha: int = 3
# k = 0.4 * h**2 / eps
# Tf = 42.0
# x = np.arange(0+h, 20-h, h)
#
#
# L = construct_laplace_matrix_1d(N, h)
# bc = np.concatenate(([1], np.zeros(N-1)))/h**2
#
# u = (x<3).astype('float64')
#
# fig, ax = plt.subplots()
# ln, = plt.plot(x, u)
# ax.set_ylim(0, 1.1)
#
# u_data = [None]
#
# for i in range(N):
#     u_data.append(u + k * (eps * (L * u + bc) + (u - u ** alpha)))
#
# t = np.linspace(0, Tf, N)
# X, T = np.meshgrid(x, t)
#
# print(len(u_data))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, T, u_data, 50)
