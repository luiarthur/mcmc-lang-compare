import numba
import numpy as np
import distributions
from tqdm import trange

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
    return metropolis_base(x, log_prob, proposal_sd)[0]


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
def log_jacobian_log_x(log_x, x):
    return log_x


@numba.njit
def log_jacobian_logit_x(logit_x, x):
    return np.log(x) + np.log1p(-x)


@numba.njit
def ametropolis_transformed_var(x, to_real, to_x, log_prob, log_jacobian, tuner):
    real_x = to_real(x)

    def lpdf_real_x(rx):
        xx = to_x(rx)
        return log_prob(xx) + log_jacobian(rx, xx)

    # From metropolis_base
    # FIXME: Currently, in numba, functions can be created
    #        in functions as long as:
    #        1. they are not passed into other functions, but used immediately.
    #        2. they are not returned as the final result
    #        Consequently, I need to repeat some code. Any workarounds here?
    real_proposal = tuner.proposal_sd * np.random.randn() + real_x
    acceptance_log_prob = lpdf_real_x(real_proposal) - lpdf_real_x(real_x)
    accept = acceptance_log_prob > np.log(np.random.rand())
    tuner.update(accept)

    if accept:
        x = to_x(real_proposal)

    return x


@numba.njit
def ametropolis_positive_var(x, log_prob, tuner):
    return ametropolis_transformed_var(x, np.log, np.exp,
                                       log_prob, log_jacobian_log_x, tuner)


@numba.njit
def ametropolis_unit_var(x, log_prob, tuner):
    return ametropolis_transformed_var(x, logit, sigmoid,
                                       log_prob, log_jacobian_logit_x, tuner)


def fieldnames(obj):
    fields = obj.__class__.__dict__.keys()
    return list(filter(lambda f: not f.startswith('__'), fields))


class Gibbs():
    def __init__(self, model, state_name='state'):
        self.state_name = state_name
        self.all_params = fieldnames(model. __getattribute__(state_name))
        self.model = model


    def fit(self, niter, nburn, track_params=None):
        if track_params is None:
            track_params = self.all_params

        out = []

        for i in trange(niter + nburn):
            self.model.update()

            if i >= nburn:
                curr = dict()
                for key in track_params:
                    curr[key] = (self.model
                                     .__getattribute__(self.state_name)
                                     .__getattribute__(key))

                out.append(curr)
        
        return out
