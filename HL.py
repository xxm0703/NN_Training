import numpy as np
from activation import *
from utils import *

__all__ = ['HL']


class HL(object):
    """
    A Neural Network with 1 Hidden Layer
    Used Logistics Regression Class for help
    """

    def __init__(self, inputs=2, hidden_units=4, classes=2, read_file=None):

        if read_file is None:
            self.layers = [Layer(inputs, hidden_units, Sigmoid), Layer(hidden_units, classes, Sigmoid)]
        else:
            self.layers = np.load(read_file)['HL']

        self.learning_rate = 0.4

    def forward(self, x):
        o = x
        for layer in self.layers:
            o = layer.forward_propagation(o)
        return o

    def backward(self, y, output):
        da = self.cost(output, y)
        for layer in self.layers:
            da = layer.back_propagation(da, self.learning_rate)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(y, o)

    @staticmethod
    def cost(output, t):
        loss = - output / t + (1 - output) / (1 - t)
        return np.sum(loss) / t.shape[0]


if __name__ == '__main__':
    model = HL()
