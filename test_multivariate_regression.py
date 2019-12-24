import numpy as np
from simple_neural_network import SimpleNeuralNetwork

if __name__ == "__main__":
    X = np.expand_dims(np.arange(10), axis=1)
    Y = np.array([[i, i+1] for i in range(10)]) * 2

    neural_network = SimpleNeuralNetwork(1, [20, 20], 2)
    learning_rate = 0.01
    num_iterations = 10000
    for i in range(num_iterations):
        loss, predictions, dW, db = neural_network.propagate(X, Y)
        for j in range(len(neural_network.W)):
            neural_network.W[j] -= learning_rate * dW[j]
            neural_network.b[j] -= learning_rate * db[j]
        if i % 1000 == 0:
            print("loss:", loss)
