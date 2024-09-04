import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    """
    Implementation of abstract class Layer as a Convolutional layer (input is cross-correlated with kernel to produce an output matrix)
    
    @param: (tuple) input_shape - height, depth and width of input
    @param: (int) kernel_size - size of each matrix inside each kernel
    @param: (int) depth - the number of kernels
    """
    def __init__(self, input_shape, kernel_size, depth, stride=1):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.depth = depth
        self.stride = stride
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # Initialise kernels and biases
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        """
        Compute the forward propagation step

        @param: (np.array) input - input matrix
        
        @return: numpy array after forward pass through layer 
        """
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(0, self.input_depth, self.stride):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Calculate partial derivatives for weights and input and update weights and biases
        """
        # Initialise arrays for kernel and input gradients
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient