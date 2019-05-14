import numba
import numpy as np
import time

import sys
sys.path.append('../')
import mcmc
import distributions as dist

@numba.njit
def lpdf_std_normal(x):
    return mcmc.lpdf_normal(x, 0, 1)

@numba.jit
def sample_std_normal(n):
    tuner = mcmc.Tuner(1.0)

    xs = np.zeros(n)
    x = 0.0
    for b in range(n):
        x = mcmc.ametropolis(x, dist.Normal(0, 1).lpdf, tuner)
        # x = mcmc.ametropolis(x, lpdf_std_normal, tuner) # faster
        xs[b] = x
    
    return xs

print('compile...')
x = sample_std_normal(1)

print('time...')
B = int(1e6)
tic = time.time()
x = sample_std_normal(B)
toc = time.time()

elapsed_time = (toc - tic)
print('elapsed time: {}'.format(elapsed_time)) # same speed as julia :)
