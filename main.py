from simple_neural_network import SimpleNeuralNetwork
import utils

if __name__ == "__main__":
    X, Y = utils.generate_random_data()
    utils.plot_data(X, Y, "output/data.png")

    neural_network = SimpleNeuralNetwork(1, [10, 10], 1)
    learning_rate = 0.01
    num_iterations = 10000
    for i in range(num_iterations):
        loss, predictions, dW, db = neural_network.propagate(X, Y)
        for j in range(len(neural_network.W)):
            neural_network.W[j] -= learning_rate * dW[j]
            neural_network.b[j] -= learning_rate * db[j]
        if i % 1000 == 0:
            print("loss:", loss)

    loss, Z_cache, A_cache = neural_network.propagate_forward(X, Y)
    utils.plot_results(X, Y, A_cache[-1], "output/results.png")
