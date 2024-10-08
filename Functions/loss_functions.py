from function import Function
import numpy as np

# Loss function does not need to be instantiated. Class only written in interest of encapsulation

class MSE(Function):
    """
    Implementation of Mean Squared Error loss function extends Function class
    """
    def __init__(self):
         pass
    
    @staticmethod
    def loss(y_true, y_pred):
        """Returns MSE for predicted and true values"""
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def derivative(y_true, y_pred):
            """Returns derivative of MSE for predicted and true values"""
            return 2 * (y_pred - y_true) / np.size(y_true)

class BinaryCrossEntropy(Function):
    """
    Implementation of Binary Cross Entropy loss function extends Function class
    """
     
    def __init__(self):
        pass
    
    @staticmethod
    def loss(y_true, y_pred):
        """Returns Binary Cross Entropy for predicted and true values"""
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        """Returns derivative of Binary Cross Entropy for predicted and true values"""
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class CategoricalCrossEntropy(Function):
    """
    Implementation of Categorical Cross Entropy loss function extends Function class
    """
     
    def __init__(self):
        pass

    @staticmethod
    def loss(y_true, y_pred):
        """Returns Categorical Cross Entropy for predicted and true values"""
        return -np.sum(y_true * np.log(y_pred + 10**-100))

    @staticmethod
    def derivative(y_true, y_pred):
        """Returns derivative of Categorical Cross Entropy for predicted and true values"""
        return -y_true / (y_pred + 10**-100)

class SSE(Function):
    """
    Implementation of Sum of Squared Errors loss function extends Function class
    """
     
    def __init__(self):
        pass

    @staticmethod
    def loss(y_pred, y_true):
        """Returns Sum of Squareed errors for predicted and true values"""
        return 0.5 * np.sum(np.power(y_true - y_pred, 2))

    @staticmethod
    def prime(y_true, y_pred):
        """Returns derivative of Sum of Squared Errors for predicted and true values"""
        return y_pred - y_true