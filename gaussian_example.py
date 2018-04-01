import torch
from torch.autograd import Variable

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

"""
Gaussian Distribution
"""

class gaussian:

    def __init__(self, params):
        self.mu = params['mu']
        self.sigma = params['sigma']

    def density(self, x, mu=None, sigma=None):
        if mu == None: mu = self.mu
        if sigma == None: sigma = self.sigma

        p_1 = (((2 * np.pi)**(len(mu)/2)) * (np.linalg.det(sigma))**0.5)
        p_2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(sigma))).dot((x-mu))

        return np.exp(p_2)/p_1

    def log_density(self, x):
        return np.log(self.density(x))

    def grad_log_density(self,x, z_optim=False):
        dtype = torch.FloatTensor

        mu = Variable(torch.Tensor(self.mu).type(dtype), requires_grad=z_optim)
        sigma = Variable(torch.Tensor(self.sigma).type(dtype), requires_grad=False)
        x = Variable(torch.Tensor(x).type(dtype), requires_grad=True)

        y = (-1/2) * torch.dot(x - mu, torch.inverse(sigma).mv(x - mu))
        y.backward()

        if z_optim:
            return dict(x_grad=x.grad, mu_grad=mu.grad)

        return x.grad.data.numpy()

    # write manual gradient here and a test
    def grad_log_density_1(self,x, z_optim=False):
        dtype = torch.FloatTensor

        mu = Variable(torch.Tensor(self.mu).type(dtype), requires_grad=z_optim)
        sigma = Variable(torch.Tensor(self.sigma).type(dtype), requires_grad=False)
        x = Variable(torch.Tensor(x).type(dtype), requires_grad=True)

        y = (-1/2) * torch.dot(x - mu, torch.inverse(sigma).mv(x - mu))
        y.backward()

        if z_optim:
            return dict(x_grad=x.grad, mu_grad=mu.grad)

        return x.grad.data.numpy()

        ###### check this grad density

    def sampler(self, N, mu=None, sigma=None):
        if mu == None: mu = self.mu
        if sigma == None: sigma = self.sigma

        # m, n = sigma.size
        # l = torch.from_numpy(np.linalg.cholesky(sigma))
        # x = torch.randn(n,)

        return multivariate_normal.rvs(mean=mu, cov=sigma, size=N)
        # return torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mu), torch.Tensor(sigma))






