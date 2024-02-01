"""
Contain the implementation of a simple neural network
Author: Son Phat Tran
"""
import numpy as np
from utils import sigmoid, sigmoid_derivative


class ConventionalNeuralNetwork:
    def __init__(self, input_size, hidden_size):
        """
        Create a two-layer neural network
        NOTE:
        - The network does not include any bias b
        - The network uses the sigmoid activation function
        :param input_size: size of the input vector
        :param hidden_size: size of the hidden layer
        :return:
        """
        # Cache the size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create the layer
        self.W1 = np.random.normal(size=(self.input_size, self.hidden_size))
        self.W2 = np.random.normal(size=(self.hidden_size, 1))

        # Create a cache
        self.cache = {}

    def forward(self, x_train, y_train):
        """
        Perform the forward pass of the neural network
        :param x_train: the training input of the neural network
        :param y_train: the training
        :return: the output of the neural network
        """
        # Calculate the output of the first layer
        a1 = x_train @ self.W1
        z1 = sigmoid(a1)

        # Calculate the output of the second layer
        a2 = z1 @ self.W2

        # Cache the values
        self.cache = {
            "x_train": x_train,
            "y_train": y_train,
            "a1": a1,
            "z1": z1,
            "a2": a2
        }

        # Calculate the error function
        score = (1 / 2) * np.sum((y_train.reshape(-1) - a2.reshape(-1)) ** 2) / y_train.shape[0]
        return a2, score

    def predict(self, x_test):
        """
        Perform the prediction
        :param x_test: the test points
        :return: the output prediction
        """
        # Calculate the output of the first layer
        a1 = x_test @ self.W1
        z1 = sigmoid(a1)

        # Calculate the output of the second layer
        a2 = z1 @ self.W2
        return a2

    def backward(self, learning_rate):
        """
        Perform back-propagation
        :param learning_rate: Learning rate of back-propagation
        :return: None
        """
        # Get cached values
        x_train, y_train, a1, z1, a2 = self.cache["x_train"], self.cache["y_train"], \
            self.cache["a1"], self.cache["z1"], self.cache["a2"]

        # Calculate the gradient w.r.t a2
        d_a2 = (a2 - y_train.reshape(-1, 1)).reshape(-1, 1)

        # Calculate the gradient w.r.t z1
        d_z1 = d_a2 @ self.W2.T

        # Calculate the gradient w.r.t W2
        d_W2 = z1.T @ d_a2

        # Calculate the gradient w.r.t a1
        d_a1 = d_z1 * sigmoid_derivative(a1)

        # Calculate the gradient w.r.t W1
        d_W1 = x_train.T @ d_a1

        # Perform back-prop
        self.W1 -= learning_rate * d_W1
        self.W2 -= learning_rate * d_W2




