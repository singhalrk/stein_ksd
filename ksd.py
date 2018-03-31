import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import seaborn as sns

from kernels import kernels

from tqdm import tqdm
"""
Kernelized Stein Discrepancy
wqdhewbfrkejb
"""

class ksd:
    def __init__(self, params):
        self.k_method = kernels(params['name'])
        self.k = self.k_method.get_kernel()
        self.grad_kx = self.k_method.grad_kx
        self.grad_ky = self.k_method.grad_ky
        self.grad_kxy = self.k_method.grad_kxy

        self.p = params['p']
        self.q = params['q']



    def h(self, x, y):

        p1 = self.k(x,y) * np.dot(self.p.grad_log_density(y), self.p.grad_log_density(x))
        p2 = np.dot(self.grad_kx(x,y), self.p.grad_log_density(y))
        p3 = np.dot(self.grad_ky(x,y), self.p.grad_log_density(x))
        p4 = np.sum(self.grad_kxy(x,y))

        return p1 + p2 + p3 + p4

    def stein_Op(self, n):
        x_samples = self.q.sampler(n)
        y_samples = self.q.sampler(n)

        stein_average = np.sum(np.array([[self.h(x_i,y_j) for x_i in x_samples] for y_j in tqdm(y_samples)]))
        # stein_average = 0
        # for x in tqdm(x_samples):
        #     stein_average += np.sum(np.array([self.h(x, y_) for y_ in y_samples]))

        return stein_average/(n**2)




