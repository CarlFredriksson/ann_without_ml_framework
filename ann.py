import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, input_layer_size, hidden_layer_sizes):
        # Limited to 1 dimensional output for now
        output_layer_size = 1
        if len(hidden_layer_sizes ) == 0:
            self.W = [ np.random.randn(input_layer_size, output_layer_size) * 0.01 ]
            self.b = [ np.zeros((1, output_layer_size)) ]
        elif len(hidden_layer_sizes ) == 1:
            self.W = [
                np.random.randn(input_layer_size, hidden_layer_sizes[0]) * 0.01,
                np.random.randn(hidden_layer_sizes[0], output_layer_size) * 0.01
            ]
            self.b = [
                np.zeros((1, hidden_layer_sizes[0])),
                np.zeros((1, output_layer_size))
            ]
        elif len(hidden_layer_sizes) == 2:
            self.W = [
                np.random.randn(input_layer_size, hidden_layer_sizes[0]) * 0.01,
                np.random.randn(hidden_layer_sizes[0], hidden_layer_sizes[1]) * 0.01,
                np.random.randn(hidden_layer_sizes[1], output_layer_size) * 0.01
            ]
            self.b = [
                np.zeros((1, hidden_layer_sizes[0])),
                np.zeros((1, hidden_layer_sizes[1])),
                np.zeros((1, output_layer_size))
            ]

    def propagate(self, X, Y):
        loss, Z_cache, A_cache = ann.propagate_forward(X, Y)
        dW, db = ann.propagate_backward(X, Y, Z_cache, A_cache)
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
                    loss1, _, _ = ann.propagate_forward(X, Y)
                    self.W[i][j][k] -= 2 * epsilon
                    loss2, _, _ = ann.propagate_forward(X, Y)
                    self.W[i][j][k] += epsilon
                    dW[i][j][k] = (loss1 - loss2) / (2 * epsilon)

        # Approximate bias derivatives
        db = [None] * len(self.b)
        for i in range(len(self.b)):
            db[i] = np.zeros(np.shape(self.b[i]))
            for j in range(np.shape(self.b[i])[0]):
                for k in range(np.shape(self.b[i])[1]):
                    self.b[i][j][k] += epsilon
                    loss1, _, _ = ann.propagate_forward(X, Y)
                    self.b[i][j][k] -= 2 * epsilon
                    loss2, _, _ = ann.propagate_forward(X, Y)
                    self.b[i][j][k] += epsilon
                    db[i][j][k] = (loss1 - loss2) / (2 * epsilon)

        return dW, db

def generate_random_data():
    X = np.expand_dims(np.linspace(0, 4, num=200), axis=1)
    Y = X**2 - 2
    noise = np.random.normal(0, 2, size=X.shape)
    Y = Y + noise
    Y = Y.astype("float32")
    return X, Y

def plot_data(X, Y, plot_path):
    plt.scatter(X, Y, color="blue")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.clf()

def plot_results(X, Y, Y_predict, plot_path):
    plt.scatter(X, Y, color="blue")
    plt.plot(X, Y_predict, color="red")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    """
    #np.random.seed(3)
    X = np.ones((5, 2)) * np.array([[1], [2], [3], [4], [5]])
    Y = np.array([[0], [1], [0.5], [1], [0]])
    ann = ANN(2, [3, 3])
    loss, predictions, dW, db = ann.propagate(X, Y)
    print("***Gradient from backprop***")
    print("dW:")
    [print(d) for d in dW]
    print()
    print("db:")
    [print(d) for d in db]
    print()
    dW_approx, db_approx = ann.compute_gradient_approximately(X, Y)
    print("***Approximate gradient for checking backprop***")
    print("dW_approx:")
    [print(d) for d in dW_approx]
    print()
    print("db_approx:")
    [print(d) for d in db]
    print()
    """
    X_train, Y_train = generate_random_data()
    X_val, Y_val = generate_random_data()
    plot_data(X_train, Y_train, "output/data_train.png")
    plot_data(X_val, Y_val, "output/data_val.png")

    ann = ANN(1, [5, 5])
    learning_rate = 0.01
    num_iterations = 10000
    for i in range(num_iterations):
        loss, predictions, dW, db = ann.propagate(X_train, Y_train)
        for j in range(len(ann.W)):
            ann.W[j] -= learning_rate * dW[j]
            ann.b[j] -= learning_rate * db[j]
        if i % 1000 == 0:
            print("loss:", loss)

    loss, Z_cache, A_cache = ann.propagate_forward(X_train, Y_train)
    plot_results(X_val, Y_val, A_cache[-1], "output/results_train.png")
    loss, Z_cache, A_cache = ann.propagate_forward(X_val, Y_val)
    plot_results(X_val, Y_val, A_cache[-1], "output/results_val.png")
