import numpy as np

__all__ = [
    'Sigmoid',
    'ReLU',
    'SoftMax'
]


class Activation:

    def activate(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(Activation):

    def activate(self, x):
        for i in range(len(x)):
            x[i] = 1 / (1 + np.exp(-x[i]))
        return x

    def derivative(self, x):
        return self.activate(x) * (1.0 - self.activate(x))


class ReLU(Activation):
    def activate(self, x):
        for i in range(len(x)):
            x[i] = max(0, x[i])
        return x

    def derivative(self, x):
        return 0 if x < 0 else 1


class LeakyReLU(Activation):
    def activate(self, x):
        for i in range(len(x)):
            x[i] = max(0.01 * x[i], x[i])
        return x

    def derivative(self, x):
        return 0.01 if x < 0 else 1


class SoftMax(Activation):
    def activate(self, x):
        for i in range(len(x)):
            x[i] = np.exp(x) / np.sum(np.exp(x))
        return x
