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
        derivative = lambda x: logistic * (1 - logistic)

        super().__init__(logistic, derivative)