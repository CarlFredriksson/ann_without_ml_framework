import numpy as np
from simple_neural_network import SimpleNeuralNetwork
import utils

if __name__ == "__main__":
    X_train, Y_train = utils.generate_random_data()
    X_val, Y_val = utils.generate_random_data()
    utils.plot_data(X_train, Y_train, "output/data_train.png")
    utils.plot_data(X_val, Y_val, "output/data_val.png")

    neural_network = SimpleNeuralNetwork(1, [10, 10])
    learning_rate = 0.01
    num_iterations = 10000
    for i in range(num_iterations):
        loss, predictions, dW, db = neural_network.propagate(X_train, Y_train)
        for j in range(len(neural_network.W)):
            neural_network.W[j] -= learning_rate * dW[j]
            neural_network.b[j] -= learning_rate * db[j]
        if i % 1000 == 0:
            print("loss:", loss)

    loss, Z_cache, A_cache = neural_network.propagate_forward(X_train, Y_train)
    utils.plot_results(X_val, Y_val, A_cache[-1], "output/results_train.png")
    loss, Z_cache, A_cache = neural_network.propagate_forward(X_val, Y_val)
    utils.plot_results(X_val, Y_val, A_cache[-1], "output/results_val.png")
