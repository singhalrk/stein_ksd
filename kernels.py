import autograd.numpy as np
from autograd import grad, jacobian

"""
Add some commong kernels
- rbf
- imq
- polynomial
 ------ find some more  ---------
"""

class kernels:
    def __init__(self, name='rbf', beta=None, c=None, degree=None):
        self.name = name
        self.beta = beta
        self.c = c
        self.degree = degree

    @staticmethod
    def rbf(x, y, beta=0.5):
        return np.exp(-((x - y)**2).sum() * beta)

    @staticmethod
    def imq(x, y, c=0.5, beta=-0.5):
        return (c + ((x - y)**2).sum())**(-beta)

    @staticmethod
    def polynomial(x, y, c=1, degree=2):
        return (c + np.dot(x,y))**degree

    # here we fix y and take a derivative wrt x
    def grad_kx(self,x,y):
        k = self.get_kernel(self.name)
        k_x = lambda x_: k(x_,y)

        return grad(k_x)(x)

    # here we fix x and take a derivative wrt y

    def grad_ky(self,x, y):
        k = self.get_kernel(self.name)
        k_y = lambda y_: k(x,y_)
        return grad(k_y)(y)

    # here we take the derivative of k wrt x and y
    def grad_kxy(self,x,y):

        kx_y = lambda y_: self.grad_kx(x, y_)

        return jacobian(kx_y)(y)


    def get_kernel(self,name=None):
        if name == None:
            name = self.name

        if name == 'rbf':
            return self.rbf

        if name == 'imq':
            return self.imq

        if name == 'polynomial':
            return self.polynomial


