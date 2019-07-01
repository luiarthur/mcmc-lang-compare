def intense(int n):
    cdef int x = 0
    cdef int i
    for i in range(n):
        x += i
        for i in range(n):
            x += i
    return x

def not_intense(n):
    x = 0
    for i in range(n):
        x += i
        for i in range(n):
            x += i
    return x

