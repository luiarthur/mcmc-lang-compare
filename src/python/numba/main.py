import numba
import numpy as np
import matplotlib.pyplot as plt
import distributions as dist
import mcmc
import Timer
from tqdm import trange

def read_simdat(path_to_simdat):
    with open(path_to_simdat, "r") as f:
        header = f.readline()
        lines = f.readlines()
        x = []
        y = []
        p = []
        for line in lines:
            xn, yn, pn = line.split(',')
            x.append(float(xn))
            y.append(float(yn))
            p.append(float(pn))
        return {'x': x, 'y': y, 'p': p}

state_spec = [('b0', numba.float64), ('b1', numba.float64), ('b2', numba.float64)]
@numba.jitclass(state_spec)
class State():
    def __init__(self, b0, b1, b2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

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
def b2_prior():
    return dist.Normal(0, 10)


@numba.njit
def log_prob_b0(b0, state, data):
    # loglike
    logit_p = b0 + state.b1 * data.x + state.b2 * data.x ** 2
    p = mcmc.sigmoid(logit_p)
    loglike = data.y * np.log(p) + (1 - data.y) * np.log1p(-p)

    # logprior
    logprior = b0_prior().lpdf(b0)

    return loglike.sum() + logprior

@numba.njit
def log_prob_b1(b1, state, data):
    # loglike
    logit_p = state.b0 + b1 * data.x + state.b2 * data.x ** 2
    p = mcmc.sigmoid(logit_p)
    loglike = data.y * np.log(p) + (1 - data.y) * np.log1p(-p)

    # logprior
    logprior = b1_prior().lpdf(b1)

    return loglike.sum() + logprior

@numba.njit
def log_prob_b2(b2, state, data):
    # loglike
    logit_p = state.b0 + state.b1 * data.x + b2 * data.x ** 2
    p = mcmc.sigmoid(logit_p)
    loglike = data.y * np.log(p) + (1 - data.y) * np.log1p(-p)

    # logprior
    logprior = b2_prior().lpdf(b2)

    return loglike.sum() + logprior


def update_b0(state, data, tuner_b0):
    log_prob = lambda b0: log_prob_b0(b0, state, data)
    state.b0 = mcmc.ametropolis(state.b0, log_prob, tuner_b0)

def update_b1(state, data, tuner_b1):
    def log_prob(b1):
        return log_prob_b1(b1, state, data)
    state.b1 = mcmc.ametropolis(state.b1, log_prob, tuner_b1)

def update_b2(state, data, tuner_b2):
    def log_prob(b2):
        return log_prob_b2(b2, state, data)
    state.b2 = mcmc.ametropolis(state.b2, log_prob, tuner_b2)


# @numba.njit
def fit(init, data, niter=1000, burn=1000, print_freq=100):
    tuner_b0 = mcmc.Tuner(.1)
    tuner_b1 = mcmc.Tuner(.1)
    tuner_b2 = mcmc.Tuner(.1)
    state = init

    out = []
    for i in trange(niter + burn):
        update_b0(state, data, tuner_b0)
        update_b1(state, data, tuner_b1)
        update_b2(state, data, tuner_b2)
        curr = {'b0': state.b0, 'b1': state.b1, 'b2': state.b2}
        if i >= burn:
            out.append(curr)
    return out


if __name__ == '__main__':
    # Path to simulated data
    path_to_simdat = '../../../dat/dat.txt'

    # Read simulated data
    simdat = read_simdat(path_to_simdat)
    x = np.array(simdat['x'])
    y = np.array(simdat['y'])
    N = x.shape[0]
    
    # COMPILE
    _ = fit(State(0, 0, 0), Data(x, y), burn=1, niter=1)

    with Timer.Timer("MCMC", digits=3):
        out = fit(State(0, 0, 0), Data(x, y), burn=1000)
    print('Done')

    b0 = np.array([s['b0'] for s in out])
    b1 = np.array([s['b1'] for s in out])
    b2 = np.array([s['b2'] for s in out])
    
    B = len(out)
    M = 200
    xx = np.linspace(-4, 4, 100)
    p = mcmc.sigmoid(b0[:, None] + b1[:, None] * xx[None, :] + b2[:, None] * xx[None, :] ** 2)

    # Plots
    plt.plot(xx, p.mean(0), label='est')
    plt.plot(xx, np.quantile(p, .975, axis=0), linestyle='--')
    plt.plot(xx, np.quantile(p, .025, axis=0), linestyle='--')
    plt.scatter(simdat['x'][::100], simdat['p'][::100], label='true', s=5)
    plt.scatter(simdat['x'], simdat['y'] + np.random.randn(len(simdat['y'])) * .01, label='data',
                s=5, alpha=.1)
    plt.legend()
    plt.show()
