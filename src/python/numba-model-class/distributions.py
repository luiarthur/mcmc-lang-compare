import numba
import numpy as np

### Normal Distributions ###
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
        return -np.log(2*np.pi)/2 - np.log(self.s) - (z**2)/2

    def mean(self):
        return self.m

    def std(self):
        return self.s

    def var(self):
        return self.s ** 2

    def sample(self):
        return np.random.normal(self.m, self.s)

    def samples(self, size):
        return np.random.normal(self.m, self.s, size=size)


### Gamma Distributions ###
gamma_spec = [
    ('shape', numba.float64),
    ('rate', numba.float64),
    ('scale', numba.float64),
]

@numba.jitclass(gamma_spec)
class Gamma():
    def __init__(self, shape, rate):
        self.shape = shape
        self.rate = rate
        self.scale = 1 / rate

    def lpdf(self, x):
        norm_const = self.shape * np.log(self.rate) - np.math.lgamma(self.shape)
        kernel = (self.shape - 1) * np.log(x) - x * self.rate
        return norm_const + kernel

    def mean(self):
        return self.shape / self.rate

    def var(self):
        return self.shape / (self.rate ** 2)

    def std(self):
        return np.sqrt(self.var())

    def sample(self):
        return np.random.gamma(self.shape, self.scale)

    def samples(self, size):
        return np.random.gamma(self.shape, self.scale, size=size)


### Inverse Gamma Distributions ###
inversegamma_spec = [
    ('shape', numba.float64),
    ('scale', numba.float64),
]

@numba.jitclass(inversegamma_spec)
class InvGamma():
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def lpdf(self, x):
        norm_const = self.shape * np.log(self.scale) - np.math.lgamma(self.shape)
        kernel = -(self.shape + 1) * np.log(x) - self.scale / x
        return norm_const + kernel

    def mean(self):
        if self.shape > 1:
            out = self.scale / (self.shape - 1)
        else:
            out = np.nan
        return out

    def var(self):
        if self.shape > 2:
            out = self.scale ** 2 / ((self.shape - 1)**2 * (self.shape - 2))
        else:
            out = np.nan
        return out

    def std(self):
        return np.sqrt(self.var())

    def sample(self):
        return 1 / np.random.gamma(self.shape, 1/self.scale)

    def samples(self, size):
        return 1 / np.random.gamma(self.shape, 1/self.scale, size=size)

### Beta Distributions ###
beta_spec = [
    ('a', numba.float64),
    ('b', numba.float64),
]

@numba.njit
def log_beta_fn(a, b):
    return np.math.lgamma(a) + np.math.lgamma(b) - np.math.lgamma(a + b) 

@numba.jitclass(beta_spec)
class Beta():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def lpdf(self, x):
        out = (self.a - 1) * np.log(x) + (self.b - 1) * np.log1p(-x) 
        out -= log_beta_fn(self.a, self.b)
        return out

    def mean(self):
        return self.a / (self.a + self.b)

    def var(self):
        return self.a * self.b / ((self.a + self.b)**2 * (self.a + self.b + 1))

    def std(self):
        return np.sqrt(self.var())

    def sample(self):
        return np.random.beta(self.a, self.b)

    def samples(self, size):
        return np.random.beta(self.a, self.b, size=size)


### Dirichlet Distribution ###
dirichlet_spec = [
    ('conc', numba.float64[:]),
]


@numba.njit
def log_beta_fn_vec(a):
    sum_lgamma_a = 0.

    for ai in a: 
        sum_lgamma_a += np.math.lgamma(ai)

    return sum_lgamma_a - np.math.lgamma(a.sum()) 


@numba.jitclass(dirichlet_spec)
class Dirichlet():
    def __init__(self, conc):
        assert np.all(conc > 0)
        self.conc = conc
    
    @property
    def size(self):
        return self.conc.size

    def lpdf(self, x):
        return (self.conc - 1) * np.log(x) - log_beta_fn_vec(self.conc)

    def mean(self):
        return self.conc / self.conc.sum()

    def var(self):
        a0 = self.conc.sum()
        a = self.conc / a0
        var_x = a * (1 - a) / (a0 + 1)

        n = self.size
        cov_mat = np.zeros((n, n))
        np.fill_diagonal(cov_mat, var_x)

        for i in range(n):
            for j in range(i):
                c = -a[i] * a[j] / (a0 + 1)
                cov_mat[i, j] = c
                cov_mat[j, i] = c

        return cov_mat

    def sample(self):
        # Dirichlet of dimension K, is the ratio between
        # each gamma and their sum.
        w = np.array([np.random.gamma(c, 1.) for c in self.conc])
        return w / w.sum()

    def samples(self, size):
        # Preallocate output. Each row is a Dirichlet.
        samps = np.empty((size, self.size))

        # Fill each row of the output matrix with a Dirichlet realization.
        for i in range(size):
            samps[i, :] = self.sample()

        return samps
