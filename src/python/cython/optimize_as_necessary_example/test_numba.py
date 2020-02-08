import numpy as np
import numba
from Timer import Timer


@numba.njit
def metropolis(curr, log_prob, step_sd):
    cand = np.random.normal(curr, step_sd)
    log_acceptance_ratio = log_prob(cand) - log_prob(curr)
    accept = log_acceptance_ratio > np.log(np.random.rand())
    out = cand if accept else curr
    return out


@numba.njit
def log_prob_std_normal(x):
    return -(x ** 2) / 2. - np.log(2 * np.pi) / 2.


@numba.njit
def test_metropolis(n):
    x = 0.
    for i in range(n):
        x = metropolis(x, log_prob_std_normal, 1.)
    return x


# Compile
test_metropolis(1 * 1)

with Timer('test_metropolis (numba)', digits=6):
    x = test_metropolis(1000 * 1000)
    print(x)
