import numba
import numpy as np
import matplotlib.pyplot as plt
import distributions as dist
import mcmc
import Timer

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

state_spec = [('b0', numba.float64), ('b1', numba.float64)]
@numba.jitclass(state_spec)
class State():
    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b1

data_spec = [('x', numba.float64[:]), ('y', numba.float64[:]), ('N', numba.int64)]
@numba.jitclass(data_spec)
class Data():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = x.shape[0]

@numba.njit
def b0_prior():
    return dist.Normal(0, 10)

@numba.njit
def b1_prior():
    return dist.Normal(0, 10)

@numba.njit
def log_prob_b0(b0, state, data):
    # loglike
    logit_p = b0 + state.b1 * data.x
    p = mcmc.sigmoid(logit_p)
    loglike = data.y * np.log(p) + (1 - data.y) * np.log1p(-p)

    # logprior
    logprior = b0_prior().lpdf(b0)

    return loglike.sum() + logprior

@numba.njit
def log_prob_b1(b1, state, data):
    # loglike
    logit_p = state.b0 + b1 * data.x
    p = mcmc.sigmoid(logit_p)
    loglike = data.y * np.log(p) + (1 - data.y) * np.log1p(-p)

    # logprior
    logprior = b1_prior().lpdf(b1)

    return loglike.sum() + logprior

def update_b0(state, data, tuner_b0):
    def log_prob(b0):
        return log_prob_b0(b0, state, data)
    state.b0 = mcmc.ametropolis(state.b0, log_prob, tuner_b0)

def update_b1(state, data, tuner_b1):
    def log_prob(b1):
        return log_prob_b1(b1, state, data)
    state.b1 = mcmc.ametropolis(state.b1, log_prob, tuner_b1)


# @numba.njit
def fit(init, data, niter=1000, burn=1000, print_freq=100):
    tuner_b0 = mcmc.Tuner(1.0)
    tuner_b1 = mcmc.Tuner(1.0)
    state = init

    out = []
    for i in range(niter + burn):
        if (i + 1) % print_freq == 0:
            print('\r{}/{}'.format(i+1, niter+burn), end='')
        if i + 1 == (niter + burn):
            print('\nDone')

        update_b0(state, data, tuner_b0)
        update_b1(state, data, tuner_b0)
        curr = {'b0': state.b0, 'b1': state.b1}
        if i >= burn:
            out.append(curr)
    return out


if __name__ == '__main__':
    # Path to simulated data
    path_to_simdat = '../../dat/dat.txt'

    # Read simulated data
    simdat = read_simdat(path_to_simdat)
    x = np.array(simdat['x'])
    y = np.array(simdat['y'])
    
    with Timer.Timer("MCMC", digits=3):
        out = fit(State(0, 0), Data(x, y))
