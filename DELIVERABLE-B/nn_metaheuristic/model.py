import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_weights = (
            input_size * hidden_size +
            hidden_size +
            hidden_size * output_size +
            output_size
        )

    def unpack(self, vector):
        i, h, o = self.input_size, self.hidden_size, self.output_size
        idx = 0

        W1 = vector[idx:idx+i*h].reshape(i, h); idx += i*h
        b1 = vector[idx:idx+h]; idx += h
        W2 = vector[idx:idx+h*o].reshape(h, o); idx += h*o
        b2 = vector[idx:idx+o]

        return W1, b1, W2, b2

    def forward(self, X, vector):
        W1, b1, W2, b2 = self.unpack(vector)
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
