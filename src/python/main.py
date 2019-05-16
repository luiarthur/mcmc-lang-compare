import numba
import numpy as np
import matplotlib.pyplot as plt
import distributions as dist
import mcmc

def read_simdat(path_to_simdat):
    with open(path_to_simdat, "r") as f:
        header = f.readline()
        lines = f.readlines()
        x = []
        y = []
        p_true = []
        for line in lines:
            xn, yn, pn = line.split(',')
            x.append(float(xn))
            y.append(float(yn))
            p_true.append(float(pn))
        return {'x': x, 'y': y, 'p_true': p_true}

model_spec = [('b0', numba.float64), ('b1', numba.float64),
              ('x', numba.float64[:]), ('y', numba.float64[:])]
@numba.jitclass(model_spec)
class Model():
    def __init__(self, b0, b1, x, y):
        self.b0 = b0
        self.b1 = b1
        self.x = x
        self.y = y

    def b0_prior(self):
        return dist.Normal(0, 10)

    def log_prob_b0(self, b0):
        # loglike
        logit_p = b0 + self.b1 * x
        p = mcmc.sigmoid(logit_p)
        loglike = y * np.log(p) + (1 - y) * np.log1p(-p)

        # logprior
        logprior = self.b0_prior().lpdf(b0)

        return loglike + logprior

    # Cant?
    # def update_b0(self):
    #     self.b0 = mcmc.ametropolis(self.b0, self.log_prob_b0, 



if __name__ == '__main__':
    # Path to simulated data
    path_to_simdat = '../../dat/dat.txt'

    # Read simulated data
    data = read_simdat(path_to_simdat)
    x = np.array(data['x'])
    y = np.array(data['y'])
    
    model = Model(0, 0, x, y)
