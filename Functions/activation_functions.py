import numpy as np
from function import Function

class Tanh(Function):
    """
    Implementation of the tanh hyperbolic function extends Function class
    """
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        derivative = lambda x: 1 - ((np.tanh(x)) ** 2)

        super().__init__(tanh, derivative)

class Logistic(Function):
    """
    Implementation of the logistic function extends Function class
    """
    def __init__(self):
        logistic = lambda x: 1 / (1 + np.exp(-x))
        derivative = lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

        super().__init__(logistic, derivative)

class RelU(Function):
    """
    Implementation of the RelU funciton extends Function class
    """
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        derivative = lambda x: 1 * (x >= 0)

        super().__init__(relu, derivative)