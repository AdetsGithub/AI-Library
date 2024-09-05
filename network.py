class Network:
    """
    Network object contains the layers of the object and defines a train and predict method to train and predict on input data
    
    @param: (array) network - array containing layers of neural network in order
    """
    def __init__(self, network):
        # Initialise network
        self.network = network

    def predict(self, input):
        """Forward propagate input to obtain prediction"""
        for layer in self.network:
            input = layer.forward(input)
        return input

    def train(self, loss_func, train_data, target, epochs = 1000, learning_rate = 0.01, verbose = False):
        """Forward propagate and backpropagate to train model"""
        for e in range(epochs):
            error = 0
            for x, y in zip(train_data, target):
                # Forward propagation of input
                output = self.predict(x)

                # Calculate error between prediction and true value
                error += loss_func.loss(y, output)

                # Backward propagation from output neuron
                grad = loss_func.derivative(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
            # Normalise error
            error /= len(train_data) 
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")