import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size):
        self.W = []
        self.b = []
        if len(hidden_layer_sizes) == 0:
            self.add_layer(input_layer_size, output_layer_size)
        elif len(hidden_layer_sizes) >= 1:
            self.add_layer(input_layer_size, hidden_layer_sizes[0])
            for i in range(1, len(hidden_layer_sizes)):
                self.add_layer(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
            self.add_layer(hidden_layer_sizes[-1], output_layer_size)

    def add_layer(self, previous_layer_size, layer_size):
        self.W.append(np.random.randn(previous_layer_size, layer_size) * 0.01)
        self.b.append(np.zeros((1, layer_size)))

    def propagate(self, X, Y):
        loss, Z_cache, A_cache = self.propagate_forward(X, Y)
        dW, db = self.propagate_backward(X, Y, Z_cache, A_cache)
        return loss, A_cache[-1], dW, db

    def propagate_forward(self, X, Y):
        batch_size = np.shape(X)[0]
        Z_cache = [None] * (len(self.W))
        A_cache = [None] * (len(self.W))

        A_prev = X
        for i in range(len(self.W)):
            Z = np.dot(A_prev, self.W[i]) + self.b[i]
            A = Z
            # If not at the output layer, apply relu
            if i < (len(self.W) - 1):
                A = np.maximum(0, Z)
            Z_cache[i] = Z
            A_cache[i] = A
            A_prev = A

        loss = 1/(2*batch_size) * np.sum((Y - A_cache[-1])**2)

        return loss, Z_cache, A_cache

    def propagate_backward(self, X, Y, Z_cache, A_cache):
        batch_size = np.shape(X)[0]
        dW = [None] * (len(self.W))
        db = [None] * (len(self.b))

        # dA is short for dL/dA etc.
        dA = A_cache[-1] - Y
        dZ = dA
        for i in reversed(range(1, len(A_cache))):
            A = A_cache[i-1]
            dW[i] = (1/batch_size) * np.dot(A.T, dZ)
            db[i] = (1/batch_size) * np.sum(dZ, axis=0)
            dA = np.dot(dZ, self.W[i].T)
            dZ = np.multiply(dA, self.relu_derivative(Z_cache[i-1]))
        dW[0] = (1/batch_size) * np.dot(X.T, dZ)
        db[0] = (1/batch_size) * np.sum(dZ, axis=0)

        return dW, db

    def relu_derivative(self, x):
        return (x > 0) * 1

    def compute_gradient_approximately(self, X, Y):
        """This function is used for checking if backpropagation is properly implemented."""
        epsilon = 10e-4

        # Approximate weight derivatives
        dW = [None] * len(self.W)
        for i in range(len(self.W)):
            dW[i] = np.zeros(np.shape(self.W[i]))
            for j in range(np.shape(self.W[i])[0]):
                for k in range(np.shape(self.W[i])[1]):
                    self.W[i][j][k] += epsilon
                    loss1, _, _ = self.propagate_forward(X, Y)
                    self.W[i][j][k] -= 2 * epsilon
                    loss2, _, _ = self.propagate_forward(X, Y)
                    self.W[i][j][k] += epsilon
                    dW[i][j][k] = (loss1 - loss2) / (2 * epsilon)

        # Approximate bias derivatives
        db = [None] * len(self.b)
        for i in range(len(self.b)):
            db[i] = np.zeros(np.shape(self.b[i]))
            for j in range(np.shape(self.b[i])[0]):
                for k in range(np.shape(self.b[i])[1]):
                    self.b[i][j][k] += epsilon
                    loss1, _, _ = self.propagate_forward(X, Y)
                    self.b[i][j][k] -= 2 * epsilon
                    loss2, _, _ = self.propagate_forward(X, Y)
                    self.b[i][j][k] += epsilon
                    db[i][j][k] = (loss1 - loss2) / (2 * epsilon)

        return dW, db
