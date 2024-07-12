import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Implementation of abstract class Layer as a Dense layer (each input neuron connects to every output neuron)
    
    @param: (int) input_size - number of neurons in the input
    @param: (int) output_size - number of neurons in the output
    """
    def __init__(self, input_size, output_size, seed=123):
        # Initialise weights and biases with random values and seed
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((output_size, input_size))
        self.bias = rng.standard_normal((output_size, 1))

    def forward(self, input):
        """
        Compute the forward propagation step

        @param: (np.array) input - input vector with dimensions n x 1
        
        @return: numpy array after forward pass through layer 
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Calculate partial derivatives for weights and input and update weights and biases
        """
        
        # Calculating partial derivates for output gradient wrt weight and output gradient wrt input gradient 
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Updating weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient