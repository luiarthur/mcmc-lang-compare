import numba
import numpy as np

import sys
sys.path.append('../')
import mcmc
import distributions as dist
import Timer

# Note that factories are required for performance
# See: https://numba.pydata.org/numba-doc/dev/user/faq.html
@numba.njit
def lpdf_std_normal(x):
    return mcmc.lpdf_normal(x, 0, 1)

@numba.njit
def lpdf_exp_1(x):
    return -x

@numba.njit
def lpdf_beta_2_3(x):
    return (2-1) * np.log(x) + (3-1) * np.log1p(-x)


@numba.jit
def sample_std_normal(n, seed=1):
    np.random.seed(seed)
    tuner = mcmc.Tuner(1.0)
    xs = np.zeros(n)
    x = 0.0
    for b in range(n):
        x = mcmc.ametropolis(x, dist.Normal(0, 1).lpdf, tuner)
        # x = mcmc.ametropolis(x, lpdf_std_normal, tuner) # faster
        xs[b] = x
    return xs

@numba.jit
def sample_exp(n, seed=1):
    np.random.seed(seed)
    tuner = mcmc.Tuner(1.0)
    xs = np.zeros(n)
    x = 1.0
    for b in range(n):
        x = mcmc.ametropolis_positive_var(x, lpdf_exp_1, tuner)
        xs[b] = x
    return xs

@numba.jit
def sample_beta(n, seed=1):
    np.random.seed(seed)
    tuner = mcmc.Tuner(1.0)
    xs = np.zeros(n)
    x = .5
    for b in range(n):
        x = mcmc.ametropolis_unit_var(x, lpdf_beta_2_3, tuner)
        xs[b] = x
    return xs


if __name__ == '__main__':
    print('compile...')
    x = sample_std_normal(1)
    x = sample_exp(1)
    x = sample_beta(1)

    thresh = 1e-2

    print('time...')
    B = int(1e6)
    with Timer.Timer('std normal', digits=3): # as fast as julia :)
        z_draws = sample_std_normal(B)
        assert abs(z_draws.mean()) < thresh
        assert abs(z_draws.std() - 1) < thresh

    with Timer.Timer('exp', digits=3):
        exp_draws = sample_exp(B)
        assert abs(exp_draws.mean() - 1) < thresh
        assert abs(exp_draws.std() - 1) < thresh

    with Timer.Timer('beta', digits=3):
        beta_draws = sample_beta(B)
        assert abs(beta_draws.mean() - .4) < thresh
        assert abs(beta_draws.std() - .2) < thresh

