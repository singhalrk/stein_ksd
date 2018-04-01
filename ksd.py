import numpy as np

from kernels import kernels, rbf_kernel, imq_kernel, poly_kernel

from tqdm import tqdm
"""
Kernelized Stein Discrepancy
"""

class ksd:
    def __init__(self, name, params=dict(beta=1e-2)):
        self.params = params
        if name == 'rbf':
            self.k_method = rbf_kernel(params)

        if name == 'imq':
            self.k_method = imq_kernel(params)

        # if name == 'poly':
        #     self.k_method = poly_kernel(params)

        self.k = self.k_method.value
        self.grad_kx = self.k_method.grad_x
        self.grad_ky = self.k_method.grad_y
        self.grad_kxy = self.k_method.grad_xy

        self.p = params['p']
        self.q = params['q']

    def h(self, x, y):
        log_px = self.p.grad_log_density(x)
        log_py = self.p.grad_log_density(y)

        p1 = self.k(x,y) * np.dot(log_py, log_px)
        p2 = np.dot(self.grad_kx(x,y), log_py)
        p3 = np.dot(self.grad_ky(x,y), log_px)
        p4 = np.sum(self.grad_kxy(x,y))

        return p1 + p2 + p3 + p4

    def Op_value(self, n, m=5):
        x_samples = self.q.sampler(n)
        y_samples = self.q.sampler(m)

        stein_average = np.sum(np.array([np.array([self.h(y_i,x_j) for y_i in y_samples]) for x_j in tqdm(x_samples)]))

        return stein_average/(n*m)

# class ksd:
#     def __init__(self, params):
#         self.k_method = kernels(params['name'])
#         self.k = self.k_method.get_kernel()
#         self.grad_kx = self.k_method.grad_kx
#         self.grad_ky = self.k_method.grad_ky
#         self.grad_kxy = self.k_method.grad_kxy

#         self.p = params['p']
#         self.q = params['q']

#     def h(self, x, y):

#         p1 = self.k(x,y) * np.dot(self.p.grad_log_density(y), self.p.grad_log_density(x))
#         p2 = np.dot(self.grad_kx(x,y), self.p.grad_log_density(y))
#         p3 = np.dot(self.grad_ky(x,y), self.p.grad_log_density(x))
#         p4 = np.sum(self.grad_kxy(x,y))

#         return p1 + p2 + p3 + p4

#     def stein_Op(self, n, m=5):
#         x_samples = self.q.sampler(n)
#         y_samples = self.q.sampler(m)

#         stein_average = np.sum(np.array([[self.h(y_i,x_j) for y_i in y_samples] for x_j in tqdm(x_samples)]))
#         # stein_average = 0
#         # for x in tqdm(x_samples):
#         #     stein_average += np.sum(np.array([self.h(x, y_) for y_ in y_samples]))

#         return stein_average/(n*m)

# class ksd_rbf:
#     def __init__(self, params=dict(beta=1e-2)):
#         self.params = params
#         self.k_method = rbf_kernel(params)
#         self.k = self.k_method.value
#         self.grad_kx = self.k_method.grad_x
#         self.grad_ky = self.k_method.grad_y
#         self.grad_kxy = self.k_method.grad_xy

#         self.p = params['p']
#         self.q = params['q']

#     def h(self, x, y):

#         p1 = self.k(x,y) * np.dot(self.p.grad_log_density(y), self.p.grad_log_density(x))
#         p2 = np.dot(self.grad_kx(x,y), self.p.grad_log_density(y))
#         p3 = np.dot(self.grad_ky(x,y), self.p.grad_log_density(x))
#         p4 = np.sum(self.grad_kxy(x,y))

#         return p1 + p2 + p3 + p4

#     def stein_Op(self, n, m=5):
#         x_samples = self.q.sampler(n)
#         y_samples = self.q.sampler(m)

#         stein_average = np.sum(np.array([[self.h(y_i,x_j) for y_i in y_samples] for x_j in tqdm(x_samples)]))

#         return stein_average/(n*m)

# class ksd_imq:
#     def __init__(self, params=dict(beta=1, c=1)):
#         self.params = params
#         self.k_method = imq_kernel(params)
#         self.k = self.k_method.value
#         self.grad_kx = self.k_method.grad_x
#         self.grad_ky = self.k_method.grad_y
#         self.grad_kxy = self.k_method.grad_xy

#         self.p = params['p']
#         self.q = params['q']

#     def h(self, x, y):
#         p1 = self.k(x,y) * np.dot(self.p.grad_log_density(y), self.p.grad_log_density(x))
#         p2 = np.dot(self.grad_kx(x,y), self.p.grad_log_density(y))
#         p3 = np.dot(self.grad_ky(x,y), self.p.grad_log_density(x))
#         p4 = np.sum(self.grad_kxy(x,y))

#         return p1 + p2 + p3 + p4

#     def stein_Op(self, n, m=5):
#         x_samples = self.q.sampler(n)
#         y_samples = self.q.sampler(m)

#         stein_average = np.sum(np.array([[self.h(y_i,x_j) for y_i in y_samples] for x_j in tqdm(x_samples)]))

#         return stein_average/(n*m)

# class ksd_poly:
#     def __init__(self, params=dict(degree=2, c=1)):
#         self.params = params
#         self.k_method = poly_kernel(params)
#         self.k = self.k_method.value
#         self.grad_kx = self.k_method.grad_x
#         self.grad_ky = self.k_method.grad_y
#         self.grad_kxy = self.k_method.grad_xy

#         self.p = params['p']
#         self.q = params['q']

#     def h(self, x, y):

#         p1 = self.k(x,y) * np.dot(self.p.grad_log_density(y), self.p.grad_log_density(x))
#         p2 = np.dot(self.grad_kx(x,y), self.p.grad_log_density(y))
#         p3 = np.dot(self.grad_ky(x,y), self.p.grad_log_density(x))
#         p4 = np.sum(self.grad_kxy(x,y))

#         return p1 + p2 + p3 + p4

#     def stein_Op(self, n, m=5):
#         x_samples = self.q.sampler(n)
#         y_samples = self.q.sampler(m)

#         stein_average = np.sum(np.array([[self.h(y_i,x_j) for y_i in y_samples] for x_j in tqdm(x_samples)]))

#         return stein_average/(n*m)

# def ksd_operator(name, params):
#     if name == 'rbf':
#         return ksd_rbf(params)

#     if name == 'imq':
#         return ksd_imq(params)

#     if name == 'poly':
#         return ksd_poly(params)
