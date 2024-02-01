"""
Utility functions
Author: Son Phat Tran
"""
import numpy as np


def sigmoid(x):
    """
    Implement the sigmoid function
    :param x: the input
    :return: the sigmoid value
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Implement the derivative of the sigmoid function
    :param x: the input
    :return: the derivative of the sigmoid function
    """
    sigmoid_value = sigmoid(x)
    return sigmoid_value * (1 - sigmoid_value)


def sigmoid_second_derivative(x):
    """
    Implement the second derivative of the sigmoid function
    :param x: the input
    :return: the second derivative of sigmoid w.r.t the input
    """
    sigmoid_value = sigmoid(x)
    return sigmoid_value * ((1 - sigmoid_value) ** 2) - (sigmoid_value ** 2) * (1 - sigmoid_value)


def relu(x):
    return x * (x >= 0).astype(float)


def relu_derivative(x):
    return (x >= 0).astype(float)