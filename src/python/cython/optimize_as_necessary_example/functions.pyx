import numpy as np


def f1(s, int n):
    cdef double x = s.b

    cdef int i, j
    for i in range(n):
        for j in range(n):
            x += np.random.randn()

    s.b = x


def f2(s, int n):
    cdef double x = s.b

    cdef int i, j
    for i in range(n):
        for j in range(n):
            x += 1

    s.b = x


def f3(s, n):
    x = s.b

    for i in range(n):
        for j in range(n):
            x += 1

    s.b = x
