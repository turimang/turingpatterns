from create_initial_matrix import create_initial_matrix
from copying import turing_pattern
from create_data import create_data
from optimising import optimise
from MCMC import MCMC_main, MCMC_save_plot


if __name__ == '__main__':
    size = 100 # size of matrix dimensions
    time_lst = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] # times that we might want to solve for

    create_initial_matrix(size) # create initial matrix
    turing_pattern(a=2.8e-4, b=5e-3, tau=2, k=0) # making clean data
    create_data(time_lst) # making noisy data
    optimise(time=1.0, a_initial=0.0, a_final=1e-3, b_initial=0.0, b_final=1e-2, iters=10) # rough parameter recovery
    res, final_guess = MCMC_main(a_init=3e-4, b_init=5e-3, a_max=1e-3, b_max=1e-2, iters=1000) # this is quite slow
    MCMC_save_plot(res)

