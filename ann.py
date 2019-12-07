import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, input_size, hidden_size):
        # Limited to 1 dimensional output for now
        output_size = 1
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def propagate(self, X, Y):
        loss, forward_cache = ann.propagate_forward(X, Y)
        gradient = ann.propagate_backward(X, Y, forward_cache)
        return loss, forward_cache, gradient
    
    def propagate_forward(self, X, Y):
        batch_size = np.shape(X)[0]
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Z2
        loss = 1/(2*batch_size) * np.sum((Y - A2)**2)
        forward_cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
        return loss, forward_cache
    
    def propagate_backward(self, X, Y, forward_cache):
        batch_size = np.shape(Y)[0]
        Z1, A1, Z2, A2 = forward_cache["Z1"], forward_cache["A1"], forward_cache["Z2"], forward_cache["A2"]

        # dA2 is short for dL/dA2 etc.
        dA2 = A2 - Y
        dZ2 = dA2
        dW2 = (1/batch_size) * np.dot(A1.T, dZ2)
        db2 = (1/batch_size) * np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = np.multiply(dA1, self.relu_derivative(Z1))
        dW1 = (1/batch_size) * np.dot(X.T, dZ1)
        db1 = (1/batch_size) * np.sum(dZ1, axis=0)

        partial_derivatives = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return partial_derivatives
    
    def relu_derivative(self, x):
        return (x > 0) * 1
    
    def compute_gradient_approximately(self, X, Y):
        """This function is used for checking if backpropagation is properly implemented."""
        epsilon = 10e-4

        # Approximate dW1
        dW1 = np.zeros(np.shape(self.W1))
        for i in range(np.shape(dW1)[0]):
            for j in range(np.shape(dW1)[1]):
                self.W1[i][j] += epsilon
                loss1, _ = ann.propagate_forward(X, Y)
                self.W1[i][j] -= 2 * epsilon
                loss2, _ = ann.propagate_forward(X, Y)
                self.W1[i][j] += epsilon
                dW1[i][j] = (loss1 - loss2) / (2 * epsilon)

        # Approximate db1
        db1 = np.zeros(np.shape(self.b1))
        for i in range(np.shape(db1)[0]):
            for j in range(np.shape(db1)[1]):
                self.b1[i][j] += epsilon
                loss1, _ = ann.propagate_forward(X, Y)
                self.b1[i][j] -= 2 * epsilon
                loss2, _ = ann.propagate_forward(X, Y)
                self.b1[i][j] += epsilon
                db1[i][j] = (loss1 - loss2) / (2 * epsilon)

        # Approximate dW2
        dW2 = np.zeros(np.shape(self.W2))
        for i in range(np.shape(dW2)[0]):
            for j in range(np.shape(dW2)[1]):
                self.W2[i][j] += epsilon
                loss1, _ = ann.propagate_forward(X, Y)
                self.W2[i][j] -= 2 * epsilon
                loss2, _ = ann.propagate_forward(X, Y)
                self.W2[i][j] += epsilon
                dW2[i][j] = (loss1 - loss2) / (2 * epsilon)

        # Approximate db2
        db2 = np.zeros(np.shape(self.b2))
        for i in range(np.shape(db2)[0]):
            for j in range(np.shape(db2)[1]):
                self.b2[i][j] += epsilon
                loss1, _ = ann.propagate_forward(X, Y)
                self.b2[i][j] -= 2 * epsilon
                loss2, _ = ann.propagate_forward(X, Y)
                self.b2[i][j] += epsilon
                db2[i][j] = (loss1 - loss2) / (2 * epsilon)

        partial_derivatives = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return partial_derivatives

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
    X = np.ones((5, 3)) * np.array([[1], [2], [3], [4], [5]])
    Y = np.array([[0], [1], [0.5], [1], [0]])
    ann = ANN(3, 2)
    loss, forward_cache, partial_derivatives = ann.propagate(X, Y)
    print("A2:")
    print(forward_cache["A2"])
    print()
    print("loss:")
    print(loss)
    print()
    print("***Gradient from backprop***")
    print("dW1:")
    print(partial_derivatives["dW1"])
    print()
    print("db1:")
    print(partial_derivatives["db1"])
    print()
    print("dW2:")
    print(partial_derivatives["dW2"])
    print()
    print("db2:")
    print(partial_derivatives["db2"])
    print()
    partial_derivatives_approx = ann.compute_gradient_approximately(X, Y)
    print("***Approximate gradient for checking backprop***")
    print("dW1:")
    print(partial_derivatives_approx["dW1"])
    print()
    print("db1:")
    print(partial_derivatives_approx["db1"])
    print()
    print("dW2:")
    print(partial_derivatives_approx["dW2"])
    print()
    print("db2:")
    print(partial_derivatives_approx["db2"])
    """
    X_train, Y_train = generate_random_data()
    X_val, Y_val = generate_random_data()
    plot_data(X_train, Y_train, "output/data_train.png")
    plot_data(X_val, Y_val, "output/data_val.png")

    ann = ANN(1, 5)
    learning_rate = 0.1
    num_iterations = 10000
    for i in range(num_iterations):
        loss, forward_cache, partial_derivatives = ann.propagate(X_train, Y_train)
        ann.W1 -= learning_rate * partial_derivatives["dW1"]
        ann.b1 -= learning_rate * partial_derivatives["db1"]
        ann.W2 -= learning_rate * partial_derivatives["dW2"]
        ann.b2 -= learning_rate * partial_derivatives["db2"]
        if i % 1000 == 0:
            print("loss:", loss)

    loss, forward_cache = ann.propagate_forward(X_train, Y_train)
    plot_results(X_val, Y_val, forward_cache["A2"], "output/results_train.png")
    loss, forward_cache = ann.propagate_forward(X_val, Y_val)
    plot_results(X_val, Y_val, forward_cache["A2"], "output/results_val.png")
