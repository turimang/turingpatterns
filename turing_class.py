import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class NumericalSolver():
    def __init__(self):
        self.eps = 1    # Multiplier unique to specific biological systems
        self.h = 0.05   # Spacial step
        self.alpha = 3.0    # Exponent
        self.k = 0.4 * self.h ** 2 / self.eps   # Time step
        self.Tf = 42.0  # Total time frame
        self.x = np.arange(-15, 15, self.h) # 1D spacial coordinate
        self.N = len(self.x)    # Size of x-space
        self.bc = np.concatenate(([1], np.zeros(self.N-1)))/self.h**2   # Application of the Neumann boundary conditions
        self.x_anal = np.linspace(-15, 15, 10)  # 1D spacial coordinate for the scatter plot of the analytical solution

    def Analytical_Solution_Special(self, x, t):
        """
        Function that generates the analytical solution for the specialised Fischer Equation
        :param alpha: Exponent in the specialised Fischer equation
        :param x: x space
        :param t: time
        :return: Analytical solution of the specialised Fischer equation with respect to x at time t.
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
        """
        :return: Generates an array of numerical solutions to the specialised Fischer equation
        """
        for i in range(int(self.Tf / self.k)):
            u_new = u + self.k*(self.eps * (L * u + self.bc) + u*(1 - u**self.alpha))
            out.append([u_new])
            u[:] = u_new
        return u, out

    def solve_turing(self):
        """
        :return: Solves the specialised Fischer Equation given the default parameters
        """
        L = NumericalSolver().construct_laplace_matrix_1d()
        u = NumericalSolver().Analytical_Solution_Special(self.x, 0)
        out = []

        u, out = NumericalSolver().get_sols(u, L, out)
        out.append([u])
        return u, self.x, out


    def plot_fig(self):
        """
        :return: Generates plot comparing the numerical and analytical solutions to the specialised Fischer equation, comparing solutions at t=0, t=2 and t=4.
        Second plot compares the squared error between the analytical and numerical solution at t=0, t=2 and t=4 as a function of x.
        """
        u, x, out = NumericalSolver().solve_turing()

        #Analytical solution at t=0, t=2 and t=4 respectively
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

        # Initialise arrays
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

# a = NumericalSolver()
# a.plot_fig()
