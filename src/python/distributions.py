import numba
import numpy as np

normal_spec = [
    ('m', numba.float64),
    ('s', numba.float64),
]

@numba.jitclass(normal_spec)
class Normal():
    def __init__(self, m, s):
        self.m = m
        self.s = s

    def lpdf(self, x):
        z = (x - self.m) / self.s
        return -0.5 * np.log(2*np.pi) - np.log(self.s) - 0.5 * z**2

