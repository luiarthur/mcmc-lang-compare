import functions
from State import State
import numpy as np
from Timer import Timer
import mcmc


s1 = State(np.random.randn(3), 1.0)
with Timer(digits=6):
    functions.f1(s1, 1000)


with Timer(digits=6):
    np.random.randn(100 * 100).sum()


with Timer(digits=6):
    mcmc.test_metropolis(1000 * 1000)
