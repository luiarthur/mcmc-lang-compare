import numba
import numpy as np
import mcmc
import distributions
from tqdm import trange

log_prob = lambda x: distributions.Normal(0, 1).lpdf(x)

def f(x, n=1e6):
    for i in trange(int(n)):
        x = mcmc.metropolis(x, log_prob, 1.0)
    print(x)

if __name__ == '__main__':
    f(1.0, 1)
    f(1.0)
