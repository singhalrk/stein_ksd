import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
"""
Input - two distributions, p & q, as an input.
        which are themselves objects so we can sample
        from q and compute the score functions of
        both p and q.
"""

class stein_LS:
    def __init__(self, params):
        self.p = params['p']
        self.q = params['q']

    def ls_operator(self,f,x):
        grad_f = grad(f)
        return np.dot(self.p.grad_log_density(x),f(x)) + np.sum(grad_f(x))

    def Op_value(self, n, f):
        samples = self.q.sampler(n)
        stein_ls = np.array([(self.ls_operator(f,x_)) for x_ in tqdm(samples)])

        return np.sum(stein_ls)/n


