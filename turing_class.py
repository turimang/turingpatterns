import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class NumericalSolver():
    def __init__(self):
        self.eps = 1
        self.h = 0.05
        self.alpha = 3.0
        self.k = 0.4 * self.h ** 2 / self.eps
        self.Tf = 42.0
        self.x = np.arange(-15, 15, self.h)
        self.N = len(self.x)
        self.bc = np.concatenate(([1], np.zeros(self.N-1)))/self.h**2
        self.x_anal = np.linspace(-15, 15, 10)


    def Analytical_Solution_Special(self, x, t):
        """

        :param alpha: Exponent in the specialised Fischer equation
        :param x: x space
        :param t: time
        :return: Analytical solution of the specialised Fischer equation
        """
        return (-(1/2)*np.tanh(self.alpha/(2*(2*self.alpha+4)**(1/2))*(x-((self.alpha+4)*t)/((2*self.alpha + 4)**(1/2))))+1/2)**(2/self.alpha)

    def construct_laplace_matrix_1d(self):
        """
            Function generates an NxN 1D laplace matrix
            :param N: Number of iterations
            :param h: Spacial step
            :return: N x N 1D laplace matrix
            """
        e = np.ones(self.N)
        diagonals = [e, -2*e, e]
        offsets = [-1,0,1]
        L = scipy.sparse.spdiags(diagonals, offsets, self.N, self.N) / self.h**2
        return L

    def get_sols(self, u, L, out):
        for i in range(int(self.Tf / self.k)):
            u_new = u + self.k*(self.eps * (L * u + self.bc) + u*(1 - u**self.alpha))
            out.append([u_new])
            u[:] = u_new
            #u[0] = NumericalSolver().Analytical_Solution_Special(self.x, 0)[0]
        return u, out

    def solve_turing(self):
        L = NumericalSolver().construct_laplace_matrix_1d()
        u = NumericalSolver().Analytical_Solution_Special(self.x, 0)
        out = []

        fig, ax = plt.subplots()
        ax.set_ylim(0, 1.1)

        u, out = NumericalSolver().get_sols(u, L, out)
        out.append([u])
        return u, self.x, out


    def plot_fig(self):
        u, x, out = NumericalSolver().solve_turing()

        u_analstart = NumericalSolver().Analytical_Solution_Special(self.x_anal, 0)
        u_analhalf = NumericalSolver().Analytical_Solution_Special(self.x_anal, 2)
        u_analend = NumericalSolver().Analytical_Solution_Special(self.x_anal, 4)

        plt.figure()
        plt.plot(x, out[0][0], label='Numerical t=0', color='blue')
        plt.plot(x, out[int(len(out) / self.Tf) * 2][0], label='Numerical t=2', color='green')
        plt.plot(x, out[int(len(out) / self.Tf) * 4][0], label='Numerical t=4', color='red')
        plt.scatter(self.x_anal, u_analstart, label='Analytical t=0', color='blue')
        plt.scatter(self.x_anal, u_analhalf, label='Analytical t=2', color='green')
        plt.scatter(self.x_anal, u_analend, label='Analytical t=4', color='red')
        plt.xlabel('x')
        plt.ylabel('Solution')
        plt.legend()
        plt.show()

        u_anal0_error = NumericalSolver().Analytical_Solution_Special(x, 0)
        u_anal2_error = NumericalSolver().Analytical_Solution_Special(x, 2)
        u_anal4_error = NumericalSolver().Analytical_Solution_Special(x, 4)

        error0 = []
        error2 = []
        error4 = []

        for i in range(len(x)):
            error0.append((u_anal0_error[i] - out[0][0][i]) ** 2)
            error2.append((u_anal2_error[i] - out[int(len(out) / self.Tf) * 2][0][i]) ** 2)
            error4.append((u_anal4_error[i] - out[int(len(out) / self.Tf) * 4][0][i]) ** 2)

        # plt.figure()
        plt.plot(x, error0, label='Error t=0', color='blue')
        plt.plot(x, error2, label='Error t=2', color='green')
        plt.plot(x, error4, label='Error t=4', color='red')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.show()

a = NumericalSolver()
a.plot_fig()
