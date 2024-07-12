import numpy as np
from layer import Layer

class Function(Layer):
    """
    Implementation of abstract class Layer as a Function layer (each input neuron passes input through a specified function which produces an output)

    @param: (object) function - any function evaluated at a specific value
    @param: (object) derivative - the derivative of the function at that value
    """
    def __init__(self, function, derivative):
        # Initialise function function and derivate of the function
        self.function = function
        self.derivative = derivative

    def forward(self, input):
        """Applies function to input"""
        self.input = input
        return self.function(self.input)

    def backward(self, output_gradient, learning_rate):
        """Calculate input gradient using the Hadamard product of the output gradient and the derivative of the function""" 
        return np.multiply(output_gradient, self.derivative(self.input))