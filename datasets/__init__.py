"""
The file contains the code for generating a sinusoidal dataset
"""
import numpy as np


def generate_sinusoidal(low, high, size, scale):
    """
    Generate Gaussian-noised sinusoidal data
    :param low: low value of x
    :param high: high value of x
    :param size: the number of data to generate
    :param scale: the scale of Gaussian
    :return:
    """
    x = np.random.uniform(low=low, high=high, size=size)
    x = np.sort(x)

    # Generate t, with random Gaussian noise
    t = np.sin(x / 1.5)
    e = np.random.normal(size=t.shape[0], scale=scale)
    t = t + e

    # Reshape x
    x = x.reshape(-1, 1)

    return x, t


def generate_linear(low, high, size, scale):
    """
    Generate Gaussian-noised sinusoidal data
    :param low: low value of x
    :param high: high value of x
    :param size: the number of data to generate
    :param scale: the scale of Gaussian
    :return:
    """
    x = np.random.uniform(low=low, high=high, size=size)
    x = np.sort(x)

    # Generate t, with random Gaussian noise
    e = np.random.normal(size=x.shape[0], scale=scale)
    t = x + e

    # Reshape x
    x = x.reshape(-1, 1)

    return x, t