import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def Analytical_Solution_Special(alpha, x, t):
    return (-(1/2)*np.tanh(alpha/(2*(2*alpha + 4)**(1/2))*(x - ((alpha+4)*t)/((2*alpha + 4)**(1/2)))) + 1/2)**(2/alpha)
    # constant1 = alpha/(2.0*np.sqrt(2.0*alpha+4.0))
    # constant2 = (alpha+4.0)/(np.sqrt(2.0*alpha+4.0))
    # return (-1.0*np.tanh(constant1*(x-constant2*t)+0.5))**(2.0/alpha)


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


eps = 1.0
h = 0.05
alpha = 3.0
k = 0.4 * h ** 2 / eps
Tf = 50
x = np.arange(-15, 15, h)


def solve_turing():
    eps = 1.0
    h = 0.05
    alpha = 3.0
    k = 0.2 * h**2 / eps
    Tf = 42.0
    x = np.arange(-15, 15, h)

    N = len(x)
    L = construct_laplace_matrix_1d(N, h)
    bc = np.concatenate(([1], np.zeros(N-1)))/h**2
    out = []
    u = Analytical_Solution_Special(alpha, x, 0)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1.1)


    def getsols():
        for i in range(int(Tf/k)):
            u_new = u + k*(eps*(L*u + bc) + u*(1-u**alpha))
            out.append([u_new])
            u[:] = u_new
            u[0] = Analytical_Solution_Special(alpha, x, 0)[0]
        return u

    u = getsols()
    out.append([u])
    print(len(out))
    return u, x, out



#
u, x, out = solve_turing()

x_anal = np.linspace(-15, 15, 10)

u_analstart = Analytical_Solution_Special(alpha, x_anal, 0)
u_analhalf = Analytical_Solution_Special(alpha, x_anal, 2)
u_analend = Analytical_Solution_Special(alpha, x_anal, 4)

plt.figure()
plt.plot(x, out[0][0], label='Numerical t=0', color='blue')
plt.plot(x, out[3360][0], label='Numerical t=2', color='green')
plt.plot(x, out[6720][0], label='Numerical t=4', color='red')
plt.scatter(x_anal, u_analstart, label='Analytical t=0', color='blue')
plt.scatter(x_anal, u_analhalf, label='Analytical t=2', color='green')
plt.scatter(x_anal, u_analend, label='Analytical t=4', color='red')
plt.xlabel('x')
plt.ylabel('Solution')
plt.legend()
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
