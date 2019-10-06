import cython
import numpy as np

from libc.stdlib cimport rand as c_rand
from libc.math cimport log, sin, sqrt


cdef double c_randn():
    return sqrt(-2. * log(c_rand())) * sin(2. * np.pi * c_rand())


ctypedef double (*double_to_double_type)(double)


cdef double log_prob_std_normal(double x):
    return -(x ** 2) / 2.


cdef double metropolis(double curr, double_to_double_type log_prob,
                       double step_sd):

    # cdef double cand = curr + c_randn() * step_sd
    cdef double cand = np.random.normal(curr, step_sd)
    cdef double log_acceptance_ratio = log_prob(cand) - log_prob(curr)
    cdef double out = curr

    # if log_acceptance_ratio > np.log(np.random.rand()):
    if log_acceptance_ratio > log(c_rand()):
        out = cand

    return out


cpdef test_metropolis(int n):
    cdef int i
    cdef double x = 0.

    for i in range(n):
        x = metropolis(x, log_prob_std_normal, 1.)

    return x
