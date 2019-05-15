import numba
import numpy as np
import distributions

### Metropolis Tuner ###
tuner_spec = [
    ('proposal_sd', numba.float64),
    ('acceptance_count', numba.int64),
    ('current_iter', numba.int64),
    ('batch_size', numba.int64),
    ('target_acceptance_rate', numba.float64),
]

@numba.jitclass(tuner_spec)
class Tuner():
    def __init__(self, proposal_sd):
        self.proposal_sd = proposal_sd
        self.acceptance_count = 1
        self.current_iter = 0
        self.batch_size = 50
        self.target_acceptance_rate = 0.44 

    def delta(self, n):
        return np.minimum(n ** (-0.5), 0.01)

    def acceptance_rate(self):
        return self.acceptance_count / self.batch_size

    def update(self, accept):
        if accept:
            self.acceptance_count += 1

        self.current_iter += 1

        if self.current_iter % self.batch_size == 0:
            n = np.floor(self.current_iter / self.batch_size)
            factor = np.exp(self.delta(n))
            if self.acceptance_rate() > self.target_acceptance_rate:
                self.proposal_sd *= factor
            else:
                self.proposal_sd /= factor
            self.acceptance_count = 0

@numba.njit
def metropolis_base(x, log_prob, proposal_sd):
    proposal = proposal_sd * np.random.randn() + x
    acceptance_log_prob = log_prob(proposal) - log_prob(x)
    accept = acceptance_log_prob > np.log(np.random.rand())
    if accept:
        x = proposal
    return x, accept
    
@numba.njit
def metropolis(x, log_prob, proposal_sd):
    return metropolis_base[0]

@numba.njit
def ametropolis(x, log_prob, tuner):
    """
    adaptive metropolis
    """
    draw, accept = metropolis_base(x, log_prob, tuner.proposal_sd)
    tuner.update(accept)
    return draw

@numba.njit
def logit(p):
    return np.log(p) - np.log1p(-p)

@numba.njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# @numba.njit(numba.float64(numba.float64, numba.float64, numba.float64))
@numba.njit
def lpdf_normal(x, m, s):
    z = (x - m) / s
    return -0.5 * np.log(2*np.pi) - np.log(s) - 0.5 * z**2

@numba.njit
def lpdf_log_x(log_x, lpdf):
    log_jacobian = log_x
    return lpdf(x) + log_jacobian

@numba.njit
def lpdf_logit_x(logit_x, lpdf):
    x = sigmoid(logit_x)
    log_jacobian = np.log(x) + np.log1p(-x)
    return lpdf(x) + log_jacobian

