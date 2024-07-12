from function import Function
import numpy as np

class MSE(Function):
    """
    Implementation of Mean Squared Error loss function
    """
    def __init__(self):
         # Loss function does not need to be instantiated. Class only written in interest of encapsulation
         pass
    
    @staticmethod
    def loss(y_true, y_pred):
        # Returns MSE for predicted and true values
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def derivative(y_true, y_pred):
            # Returns derivative of MSE for predicted and true values
            return 2 * (y_pred - y_true) / np.size(y_true)

class BinaryCrossEntropy:
    """
    Implementation of Binary Cross Entropy loss function
    """
     
    def __init__(self):
        # Loss function does not need to be instantiated. Class only written in interest of encapsulation
        pass
    
    @staticmethod
    def loss(y_true, y_pred):
        # Returns Binary Cross Entropy for predicted and true values
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        # Returns derivative of Binary Cross Entropy for predicted and true values
        return np.mean((1 - y_true) / (1 - y_pred) - y_true / y_pred)