import numpy as np
import matplotlib.pyplot as plt
from simple_neural_network import SimpleNeuralNetwork

def generate_random_data():
    X = np.expand_dims(np.linspace(-3, 3, num=200), axis=1)
    Y = X**2 - 2
    noise = np.random.normal(0, 2, size=X.shape)
    Y = Y + noise
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

def check_backprop_implementation():
    X = np.ones((5, 2)) * np.array([[1], [2], [3], [4], [5]])
    Y = np.array([[0], [1], [0.5], [1], [0]])
    neural_network = SimpleNeuralNetwork(2, [3, 4])
    loss, predictions, dW, db = neural_network.propagate(X, Y)
    print("***Gradient from backprop***")
    for i in range(len(dW)):
        print("dW[{}] shape({},{}):".format(i, np.shape(dW[i])[0], np.shape(dW[i])[1]))
        print(dW[i])
        print("db[{}]:".format(i))
        print(db[i])
    dW, db = neural_network.compute_gradient_approximately(X, Y)
    print("***Approximate gradient for checking backprop***")
    for i in range(len(dW)):
        print("dW[{}] shape({},{}):".format(i, np.shape(dW[i])[0], np.shape(dW[i])[1]))
        print(dW[i])
        print("db[{}]:".format(i))
        print(db[i])
