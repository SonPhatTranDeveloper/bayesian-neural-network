"""
Contain the implementation of a bayesian neural network
Author: Son Phat Tran
"""
import numpy as np
from utils import sigmoid, sigmoid_derivative, sigmoid_second_derivative
from datasets import generate_sinusoidal, generate_linear
import matplotlib.pyplot as plt


class BayesianNeuralNetwork:
    def __init__(self, input_size, hidden_size, alpha, beta):
        """
        Create a two-layer neural network
        NOTE:
        - The network does not include any bias b
        - The network uses the sigmoid activation function
        :param input_size: size of the input vector
        :param hidden_size: size of the hidden layer
        :param alpha: the prior precision of W
        :param beta: the likelihood precision of t
        """
        # Cache the size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Cache the prior variance
        self.alpha = alpha
        self.beta = beta

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
        score = (1 / 2) * np.sum((y_train.reshape(-1) - a2.reshape(-1)) ** 2) / y_train.shape[0] \
            + (1 / 2) * np.sum(self.W1 ** 2) + (1 / 2) * np.sum(self.W2 ** 2)

        return a2, score

    def predict(self, x_test_points, x_train, y_train):
        """
        Make full prediction, mean and variance for a datapoint
        :param x_test_points: the test datapoints
        :return: the predictions, mean and variance
        """
        # Hold the mean and variance
        means, variances = [], []

        # Get the Hessian w.r.t W1 and W2
        h_W1, h_W2 = self._get_hessian(x_train, y_train)
        h_W = np.append(h_W1, h_W2)
        A = self.alpha * np.ones(h_W1.shape[0] + h_W2.shape[0]) + self.beta * h_W

        # Go through each test points
        for x_test in x_test_points:
            #  Reshape x test
            x_test = x_test.reshape(-1, 1)

            # Calculate the mean
            y, d_W1, d_W2 = self._get_gradients_w_map(x_test)

            # Calculate the variance
            d_W = np.append(d_W1, d_W2)
            variance = (1 / self.beta) + np.dot(d_W, d_W / A)

            # Append the mean and variance
            means.append(y[0])
            variances.append(variance)

        return np.array(means).reshape(-1), np.array(variances).reshape(-1)

    def _get_gradients_w_map(self, x_test):
        """
        Get the gradient of the neural network with respect to the weights
        :param x_test: the test data point
        :return: the gradient with respect to the weights, with the prediction
        """
        # Calculate the output of the first layer
        a1 = x_test @ self.W1
        z1 = sigmoid(a1)

        # Calculate the output of the second layer
        a2 = z1 @ self.W2

        # Calculate the gradient w.r.t W1 and W2
        d_a2 = np.ones_like(a2)

        # Calculate the gradient w.r.t z1
        d_z1 = d_a2 @ self.W2.T

        # Calculate the gradient w.r.t W2
        d_W2 = z1.T @ d_a2

        # Calculate the gradient w.r.t a1
        d_a1 = d_z1 * sigmoid_derivative(a1)

        # Calculate the gradient w.r.t W1
        d_W1 = x_test.T @ d_a1
        return a2, d_W1, d_W2

    def _get_hessian(self, x_train, y_train):
        """
        Get the Hessian of the neural network w.r.t to W1 and W2
        :param x_train: Training features
        :param y_train: Training labels
        :return: the Hessian w.r.t to W1 and W2
        """
        # Perform forward propagation
        self.forward(x_train, y_train)

        # Get the w_map
        W1, W2 = self.W1, self.W2

        # Get the cached variables
        x_train, y_train, a1, z1, a2 = self.cache["x_train"], self.cache["y_train"], \
            self.cache["a1"], self.cache["z1"], self.cache["a2"]

        # Get the number of training examples
        N = x_train.shape[0]

        # Calculate the diagonal Hessian w.r.t W2
        h_W2 = np.sum(z1 ** 2, axis=0)

        # Calculate the diagonal Hessian w.r.t W1
        derivative_a = sigmoid_derivative(a1)
        second_derivative_a = sigmoid_second_derivative(a1)
        W2_repeat = np.repeat(W2.reshape(1, -1), N, axis=0)

        h_a1 = ((derivative_a ** 2) * (W2_repeat ** 2) +
                second_derivative_a * W2_repeat * (a2.reshape(-1) - y_train.reshape(-1)).reshape(-1, 1))

        h_W1 = (x_train ** 2).T @ h_a1

        # Return the Hessian w.r.t W1 and W2
        return h_W1.reshape(-1), h_W2.reshape(-1)

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
        d_a2 = self.beta * (a2 - y_train.reshape(-1, 1)).reshape(-1, 1)

        # Calculate the gradient w.r.t z1
        d_z1 = d_a2 @ self.W2.T

        # Calculate the gradient w.r.t W2
        d_W2 = z1.T @ d_a2 + self.alpha * self.W2

        # Calculate the gradient w.r.t a1
        d_a1 = d_z1 * sigmoid_derivative(a1)

        # Calculate the gradient w.r.t W1
        d_W1 = x_train.T @ d_a1 + self.alpha * self.W1

        # Perform back-prop
        self.W1 -= learning_rate * d_W1
        self.W2 -= learning_rate * d_W2


if __name__ == "__main__":
    # Create a bayesian neural network
    bnn = BayesianNeuralNetwork(input_size=1, hidden_size=1000, alpha=1, beta=20)

    # Create synthetic training data
    x, t = generate_sinusoidal(low=-10, high=10, size=100, scale=0.1)

    # Fit the neural network
    EPOCHS = 300_000
    LEARNING_RATE = 0.000004

    for epoch in range(EPOCHS):
        # Get the error and output
        output, e = bnn.forward(x, t)

        # Display the error
        print(f"EPOCHS {epoch}, Score {e}")

        # Perform back-propagation
        bnn.backward(learning_rate=LEARNING_RATE)

    # Generate the prediction
    x_test = np.arange(-10, 20, 0.01).reshape(-1, 1)
    y_test, y_var = bnn.predict(x_test, x, t)
    y_test_lower = y_test - np.sqrt(y_var)
    y_test_upper = y_test + np.sqrt(y_var)

    # Draw the prediction
    plt.plot(x_test, y_test, color='blue')
    plt.plot(x_test, y_test_lower, color='blue')
    plt.plot(x_test, y_test_upper, color='blue')
    plt.scatter(x, t, color='orange')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Sinusoidal data prediction')
    plt.show()





