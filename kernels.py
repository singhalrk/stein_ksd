import autograd.numpy as np
from autograd import grad, jacobian

"""
Add some commong kernels
- rbf
- imq
- polynomial
 ------ find some more  ---------
"""


######## make this vectorized ##########

class rbf_kernel:
    def __init__(self, params = dict(beta=1e-2)):
        self.beta = params['beta']
        assert self.beta > 0

    def value(self, x, y):
        r = ((x - y)**2).sum()
        return np.exp(-self.beta * r)

    def grad_x(self, x, y):
        r = ((x - y)**2).sum()
        return - 2 * self.beta * (x - y) * np.exp(-self.beta * r)

    def grad_y(self, x, y):
        r = ((x - y)**2).sum()
        return 2 * self.beta * (x - y) * np.exp(-self.beta * r)

    def grad_xy(self, x, y):
        assert len(x) == len(y)
        n = len(x)
        r = ((x - y)**2).sum()
        _y = 2 * self.beta * np.exp(-self.beta * r) * np.ones(n)
        _xy = 4 * self.beta**2 * (x - y)**2 * np.exp(-self.beta * r)
        return _y + _xy

class imq_kernel:
    def __init__(self, params = dict(beta = 1, c=1)):
        self.beta = params['beta']
        self.c = params['c']
        assert self.beta > 0

    def value(self, x, y):
        r = ((x - y)**2).sum()
        return 1./(self.c**2 + r)**(self.beta)

    def grad_x(self, x, y):
        r = ((x - y)**2).sum()
        return -2 * self.beta * (x - y) / (self.c**2 + r)**(self.beta + 1.)

    def grad_y(self, x, y):
        r = ((x - y)**2).sum()
        return 2 * self.beta * (x - y) / (self.c**2 + r)**(self.beta + 1.)

    def grad_xy(self, x, y):
        r = ((x - y)**2).sum()
        n = len(x)
        _y = 2 * self.beta * np.ones(n) / (self.c**2 + r)**(self.beta + 1.)
        _xy = -4 * self.beta* (self.beta + 1) * (x - y)**2 / (self.c**2 + r)**(self.beta + 2.)
        return _y + _xy

class poly_kernel:
    def __init__(self, params = dict(c=1, degree=1)):
        self.degree = params['degree']
        self.c = params['c']
        assert self.degree > 0

    def value(self, x, y):
        r = np.dot(x,y)
        return (self.c + r) ** self.degree

    def grad_x(self, x, y):
        r = np.dot(x, y)
        return self.degree * (self.c + r) ** (self.degree - 1) * y

    def grad_y(self, x, y):
        r = np.dot(x, y)
        return self.degree * (self.c + r) ** (self.degree - 1) * x

    def grad_xy(self, x, y):
        n = len(x)
        r = np.dot(x, y)
        return self.degree * (self.c + r) ** (self.degree - 1) * np.ones(n)

class kernels_1:
    def __init__(self, name='rbf', beta=None, c=None, degree=None):
        self.name = name
        self.beta = beta
        self.c = c
        self.degree = degree
        self.params = dict(beta=self.beta, degree=self.degree, c=self.c)

##  write kernels gradients manually
class kernels:
    def __init__(self, name='rbf', beta=None, c=None, degree=None):
        self.name = name
        self.beta = beta
        self.c = c
        self.degree = degree

    def rbf(self, x, y, beta=0.01):
        return np.exp(- ((x - y)**2).sum() * beta)

    def imq(self, x, y, c=1, beta=-1):
        return (c**2 + ((x - y)**2).sum())**(-beta)

    def polynomial(self, x, y, c=1, degree=2):
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




