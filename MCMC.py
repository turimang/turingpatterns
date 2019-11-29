import numpy as np
import random
from turing import turing
from calculate_error import calc_error
import os
import matplotlib.pyplot as plt
import json

class guess:
    """
    a and b parameters chosen, and the error calculated
    """
    def __init__(self, a, b, time, target):
        self.error = None
        self.a = a
        self.b = b
        self.data = turing(a, b, time)
        self.error = calc_error(self.data, target)


class proposal_dist:
    """
    A 2D proposal distribution to generate steps from each a/b parameter guess
    """
    def __init__(self):
        self.mean = [0, 0]
        self.cov = [[5e-9, 0], [0, 5e-9]]
        self.x, self.y = np.random.multivariate_normal(self.mean, self.cov, 500).T


class result_dist:
    """
    Stores the accepted N_0 and Lambda values to generate the parameter distribution
    """
    def __init__(self):
        self.a_data = []
        self.b_data = []


class decision:
    """
    Decides whether to accept or reject a guess based on the error and the ratio of errors
    """
    def accept_or_reject_1(self, accepted_guess, next_guess, results, guess_number):
        """
        :param accepted_guess: class, has a, b and error values
        :param next_guess: class, has a, b and error values
        :param results: class, stores accepted values
        :param guess_number: integer, accepted values are only stored after 1/4 of iterations have been completed
        :return: class, new or old accepted guess
        """
        if next_guess.error < accepted_guess.error:
            accepted_guess = next_guess
            results.a_data.append(accepted_guess.a)
            results.b_data.append(accepted_guess.b)
            return accepted_guess
        else:
            accepted_guess = self.accept_or_reject_2(accepted_guess, next_guess, results, guess_number)
        return accepted_guess

    def accept_or_reject_2(self, accepted_guess, next_guess, results, guess_number):
        """
        Called if new error is larger than old error.
        Decides randomly whether to accept or reject new guess, with information on the error ratio.
        """
        ratio = accepted_guess.error/(next_guess.error+accepted_guess.error)
        draw = random.uniform(0, 1)
        if draw <= ratio:
            accepted_guess = next_guess
            results.a_data.append(accepted_guess.a)
            results.b_data.append(accepted_guess.b)
        return accepted_guess

def main():
    time = 1.0
    target = np.load(os.path.join('noisy_data', str(time) + '.npy'))
    dec = decision()
    accepted_a, accepted_b = 3e-4, 5e-3 #initial guesses
    accepted_guess = guess(accepted_a, accepted_b, time, target)

    results = result_dist()
    prop = proposal_dist()
    guess_number = 1

    while guess_number < 1000:
        next_a = accepted_guess.a + random.choice(prop.x)
        next_b = accepted_guess.b + random.choice(prop.y)
        if 0.0 < next_a < 1e-3 and 0 < next_b < 1e-2:
            next_guess = guess(next_a, next_b, time, target)
            accepted_guess = dec.accept_or_reject_1(accepted_guess, next_guess, results, guess_number)
            print(guess_number, next_guess.a, accepted_guess.a, next_guess.b, accepted_guess.b)
            guess_number += 1
    return results, accepted_guess


res, final_guess = main()
json.dump(res.a_data, open(os.path.join('MCMCresults', 'a_values.json'), 'w'))
json.dump(res.b_data, open(os.path.join('MCMCresults', 'b_values.json'), 'w'))

plt.hist(res.a_data, range=(0, 1e-3))
plt.show()
plt.hist(res.b_data, range=(0, 1e-2))
plt.show()
