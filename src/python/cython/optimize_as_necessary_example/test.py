import functions
from State import State
import numpy as np
from Timer import Timer


s1 = State(np.random.randn(3), 1.0)
with Timer(digits=6):
    functions.f1(s1, 1000)


with Timer(digits=6):
    np.random.randn(100*100).sum()


