import numba
import numpy as np
import matplotlib.pyplot as plt
import distributions as dist
import mcmc
import Timer
import copy


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


state_tuner_spec = [('b0', mcmc.Tuner.class_type.instance_type),
                    ('b1', mcmc.Tuner.class_type.instance_type),
                    ('b2', mcmc.Tuner.class_type.instance_type)]
@numba.jitclass(state_tuner_spec)
class StateTuner():
    def __init__(self, init_proposal_sd):
        self.b0 = mcmc.Tuner(init_proposal_sd)
        self.b1 = mcmc.Tuner(init_proposal_sd)
        self.b2 = mcmc.Tuner(init_proposal_sd)


# The following three classes are Mixins.
# Note there is no data, i.e. `__init__()`, in this method.
class Update_b0():
    def logprob_b0(self, b0):
        return self.loglike(b0, self.state.b1, self.state.b2) + self.prior.lpdf(b0)

    def update_b0(self):
        self.state.b0 = mcmc.ametropolis(self.state.b0, self.logprob_b0, self.tuners.b0)
 

class Update_b1():
    def logprob_b1(self, b1):
        return self.loglike(self.state.b0, b1, self.state.b2) + self.prior.lpdf(b1)

    def update_b1(self):
        self.state.b1 = mcmc.ametropolis(self.state.b1, self.logprob_b1, self.tuners.b1)


class Update_b2():
    def logprob_b2(self, b2):
        return self.loglike(self.state.b0, self.state.b1, b2) + self.prior.lpdf(b2)

    def update_b2(self):
        self.state.b2 = mcmc.ametropolis(self.state.b2, self.logprob_b2, self.tuners.b2)
  

model_spec = [('state', State.class_type.instance_type),
              ('data', Data.class_type.instance_type),
              ('prior', dist.Normal.class_type.instance_type),
              ('tuners', StateTuner.class_type.instance_type)]

@numba.jitclass(model_spec)
class Model(Update_b0, Update_b1, Update_b2):
    def __init__(self, init, data, init_proposal_sd):
        self.state = init
        self.data = data
        self.prior = dist.Normal(0, 10)
        self.tuners = StateTuner(init_proposal_sd)
    

    def update(self):
        self.update_b0()
        self.update_b1()
        self.update_b2()


    def loglike(self, b0, b1, b2):
        logit_p = b0 + self.data.x * b1 + (self.data.x**2) * b2
        p = mcmc.sigmoid(logit_p)
        ll = self.data.y * np.log(p) + (1 - self.data.y) * np.log1p(-p)
        return ll.sum()
 

    def fit(self, niter=1000, nburn=1000, print_freq=100):
        out = []

        for i in range(niter + nburn):
            if (i + 1) % print_freq == 0:
                print(i + 1, '/', (niter + nburn))

            self.update()
            curr = {'b0': self.state.b0, 'b1': self.state.b1, 'b2': self.state.b2}

            if i >= nburn:
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
    model = Model(State(0, 0, 0), Data(x, y), 0.1)
    _ = model.fit(1, 1, 10)

    with Timer.Timer("MCMC", digits=3):
        out = model.fit(1000, 1000, 100)
    print('Done')

    b0 = np.array([s['b0'] for s in out])
    b1 = np.array([s['b1'] for s in out])
    b2 = np.array([s['b2'] for s in out])
    
    B = len(out)
    M = 200
    xx = np.linspace(-4, 4, 100)
    p = mcmc.sigmoid(b0[:, None] + b1[:, None] * xx[None, :]
                     + b2[:, None] * xx[None, :] ** 2)

    # Plots
    plt.plot(xx, p.mean(0), label='est')
    plt.plot(xx, np.quantile(p, .975, axis=0), linestyle='--')
    plt.plot(xx, np.quantile(p, .025, axis=0), linestyle='--')
    plt.scatter(simdat['x'][::100], simdat['p'][::100], label='true', s=5)
    plt.scatter(simdat['x'],
                simdat['y'] + np.random.randn(len(simdat['y'])) * .01,
                label='data', s=5, alpha=.1)
    plt.legend()
    plt.show()

# NOTE: This version takes 5.537s, while Julia takes 4.16 on my laptop:
# 
# CPU: 8  Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
# Memory: 7.67529 GB

# LIMITATIONS:
# - slightly slower than Julia
# - `fit` method can't have any python objects. So, printing
#    time stamps can be a pain.
# - can't use reflection with the jit-compiled `model.state.
#   So, can't access the parameters in `state` by str fieldname.
