import numpy as np

__all__ = {
    'sigmoid',
    'relu',
    'sigmoid_derivative',
    'softmax'
}


def sigmoid(x):
    for i in range(len(x)):
        x[i] = 1 / (1 + np.exp(-x[i]))
        return x


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    for i in range(len(x)):
        x[i] = max(0, x[i])
    return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
