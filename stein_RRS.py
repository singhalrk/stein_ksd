import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import seaborn as sns

from kernels import kernels

class stein_RRS:
    def __init__(self, params):
        self.p = params['p']
        self.q = params['q']
