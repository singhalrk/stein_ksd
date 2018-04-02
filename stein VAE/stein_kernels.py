import torch
from torch.autograd import Variable


"""
Radial Basis Function
input :
 x - torch.Tensor
 y - torch.Tensor
 beta - float , default is 1e-2
 output - float ?

"""
class rbf:
    def __init__(self, beta=1e-2):
        self.beta = beta

    def value(self, x, y):
        r = torch.sum((x - y)**2)
        return torch.exp(-self.beta * r)

    def rbf_x(self, x, y):
        r = torch.sum((x - y)**2)
        return 2 * self.beta * (x - y) * torch.exp(-self.beta * r)

    def rbf_y(self, x, y):
        r = torch.sum((x - y)**2)
        return -2 * self.beta  * (x - y) * torch.exp(-self.beta * r)

    def rbf_xy(self, x, y):
        r = torch.sum((x - y)**2)
        n = len(x)
        _y = n * 2 * self.beta * torch.exp(-self.beta * r)
        _xy = torch.sum( -4 * self.beta * torch.exp(-self.beta * r) * (x - y)**2 )
        return _y + _xy


class imq:
    def __init__(self, c=1, beta=1):
        self.beta = beta

    def value(self, x, y):
        r = torch.sum((x - y)**2)
        return (self.c + r).pow(-self.beta)

    def grad_x(self, x, y):
        r = torch.sum((x - y)**2)
        return -2 * self.beta * (x - y) * (self.c + r).pow(-self.beta - 1.)

    def grad_y(self, x, y):
        r = torch.sum((x - y)**2)
        return 2 * self.beta * (x - y) * (self.c + r).pow(-self.beta - 1.)

    def grad_xy(self, x, y):
        r = torch.sum((x - y)**2)
        n = len(x)

        _y = n * 2 * self.beta * (self.c + r).pow(-self.beta - 1.)
        _xy = -4 * self.beta * (self.beta + 1) * ((x - y)**2) * (self.c + r).pow(-self.beta - 2.)

        return _y + torch.sum(_xy)

