import numpy as np
from layer import Layer

class Reshape(Layer):
    """
    Implementation of abstract class Layer as a Reshape layer (reshapes input and output layer)
    
    @param: (tuple) input_shape - shape of neurons in the input
    @param: (tuple) output_shape - shape of neurons in the output
    """
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        """
        Reshape the input during forward propagation
        """
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        """
        Reshape the output during backward propagation
        """
        return np.reshape(output_gradient, self.input_shape)