import numpy as np
from layer import Layer

class RNN(Layer):
    def __init__(self, preprocessing_obj, hidden_size, seed=123):
        self.prp = preprocessing_obj
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        # Weight matrices (multiplied by 0.01 to prevent vanishing/exploding gradients)
        self.Wh = rng.standard_normal((hidden_size, hidden_size)) * 0.01  # Hidden state weight
        self.Wx = rng.standard_normal((self.prp.vocab_size, hidden_size)) * 0.01   # Input weight
        self.Wy = rng.standard_normal((hidden_size, self.prp.vocab_size)) * 0.01   # Output weight

        # Biases
        self.bh = np.zeros((1, hidden_size))  # Bias for hidden state
        self.by = np.zeros((1, self.prp.vocab_size))   # Bias for output

        # Hidden state
        self.h = np.zeros((1, hidden_size))  # Initial hidden state

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def step_forward(self, x, h_prev):
        h_current = np.tanh(np.dot(x, self.Wx) + np.dot(h_prev, self.Wh) + self.bh)
        y = np.dot(h_current, self.Wy) + self.by
        y = self.softmax(y)  # Apply softmax to the output
        return h_current, y

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        outputs = np.zeros((batch_size, seq_len, self.prp.vocab_size))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_size))
        
        h_prev = np.zeros((batch_size, self.hidden_size))  # Initial hidden state
        
        for t in range(seq_len):
            x = np.zeros((batch_size, self.prp.vocab_size))
            for i in range(batch_size):
                x[i, inputs[i, t]] = 1  # One-hot encoding of input characters
            h_current, y = self.step_forward(x, h_prev)
            outputs[:, t, :] = y
            hidden_states[:, t, :] = h_current
            h_prev = h_current  # Update hidden state for next time step
        
        return outputs, hidden_states

    def compute_loss(self, predictions, targets):
        loss = 0
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                loss -= np.log(predictions[i, j, targets[i, j]] + 1e-9)
        return loss / np.prod(targets.shape)

    def backward(self, inputs, hidden_states, outputs, targets, learning_rate=0.001):
        batch_size, seq_len = inputs.shape
        
        # Initialize gradients
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((batch_size, self.hidden_size))  # Gradient of the next hidden state
        
        for t in reversed(range(seq_len)):
            x_t = np.zeros((batch_size, self.prp.vocab_size))
            for i in range(batch_size):
                x_t[i, inputs[i, t]] = 1  # One-hot encoding of inputs
            
            h_t = hidden_states[:, t, :]
            h_prev = hidden_states[:, t - 1, :] if t > 0 else np.zeros_like(h_t)

            # Compute gradient of loss w.r.t. output
            dy = outputs[:, t, :]
            for i in range(batch_size):
                dy[i, targets[i, t]] -= 1  # Gradient for softmax
            
            dWy += np.dot(h_t.T, dy)
            dby += np.sum(dy, axis=0, keepdims=True)

            # Compute gradient of loss w.r.t. hidden state
            dh = np.dot(dy, self.Wy.T) + dh_next
            dtanh = (1 - h_t ** 2) * dh

            dWx += np.dot(x_t.T, dtanh)
            dWh += np.dot(h_prev.T, dtanh)
            dbh += np.sum(dtanh, axis=0, keepdims=True)

            dh_next = np.dot(dtanh, self.Wh.T)

        # Update weights using gradients
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(self, prompt='a', learning_rate=0.001, n_epochs=10, verbose=False):
        for epoch in range(n_epochs):
            loss = 0
            for batch_idx in range(self.prp.inputs.shape[0]):
                batch_inputs = self.prp.inputs[batch_idx]
                batch_targets = self.prp.targets[batch_idx]
                outputs, hidden_states = self.forward(batch_inputs)
                loss += self.compute_loss(outputs, batch_targets)
                self.backward(batch_inputs, hidden_states, outputs, batch_targets, learning_rate)
            # Example of generating text
            if verbose:
                print(f'\nEpoch {epoch+1}/{n_epochs}, Loss: {loss/self.prp.inputs.shape[0]:.4f}')
                generated_text = self.sample(prompt)
                print(generated_text)

    def sample(self, prompt, length=500):

        # Set the RNN to use the initial hidden state
        hidden_state = np.zeros((1, self.hidden_size))

        # Convert the seed character to an index
        input_idx = self.prp.token_to_idx[prompt]
        
        # Generate one-hot encoding for the seed character
        input_vec = np.zeros((1, self.prp.vocab_size))
        input_vec[0, input_idx] = 1

        generated_text = prompt  # Initialize the generated text with the seed character

        for _ in range(length):
            # Perform a forward step to get the prediction
            hidden_state, output = self.step_forward(input_vec, hidden_state)

            # Sample from the output probabilities
            output_prob = output[0]
            sampled_idx = np.random.choice(range(self.prp.vocab_size), p=output_prob)
            
            # Convert the sampled index back to a character
            sampled_token = self.prp.idx_to_token[sampled_idx]

            # Append the sampled character to the generated text
            generated_text += sampled_token

            # Update input to the next character (one-hot encoding)
            input_vec = np.zeros((1, self.prp.vocab_size))
            input_vec[0, sampled_idx] = 1

        return generated_text

