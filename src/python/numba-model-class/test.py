import numba
import numpy as np
import mcmc
import distributions
from tqdm import trange
from Timer import Timer

# @numba.njit
# def log_prob(x):
#     return distributions.Normal(0, 1).lpdf(x)

@numba.jit
def f(x, n):
    log_prob = distributions.Normal(0, 1).lpdf
    for _ in range(n):
        x = mcmc.metropolis(x, log_prob, 1.0)
    return x

if __name__ == '__main__':
    # Compile
    x = f(1.0, 1)

    with Timer(digits=6):
        print('x = {}'.format(f(1.0, int(2e6))))
