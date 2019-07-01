import numpy as np

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


def log_beta_fn(a, b):
    return np.math.lgamma(a) + np.math.lgamma(b) - np.math.lgamma(a + b) 

class Beta():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def lpdf(self, x):
        return (self.a - 1) * np.log(x) + (self.b - 1) * np.log1p(-x) - log_beta_fn(self.a, self.b)

    def mean(self):
        return self.a / (self.a + self.b)

    def var(self):
        return self.a * self.b / ((self.a + self.b)**2 * (self.a + self.b + 1))

    def std(self):
        return np.sqrt(self.var())
