import numpy as np
from activation import sigmoid, sigmoid_derivative

__all__ = ['HL',
           'Layer']


class Layer(object):
    def __init__(self, inputs, classes):
        self.w = np.random.randn(classes, inputs)
        self.b = np.zeros([classes, 1])
        self.a = None
        self.z = None
        self.dz = None
        self.dw = None
        self.db = None


class HL(object):
    """
    A Neural Network with 1 Hidden Layer
    Used Logistics Regression Class for help
    """

    def __init__(self, inputs=2, hidden_units=4, classes=2, read_file=None):

        if read_file is None:
            self.layers = [Layer(inputs, hidden_units), Layer(hidden_units, classes)]
        else:
            self.layers = np.load(read_file)['HL']

        self.learning_rate = 0.4

    def forward(self, x):
        output = x.T
        for l in self.layers:
            l.z = np.matmul(l.w, output) + l.b
            l.a = sigmoid(l.z)
            output = l.a
        return output

    def backward(self, x, y, o):
        m = x.shape[0]
        self.layers[1].dz = o - y
        self.layers[1].dw = np.matmul(self.layers[1].dz, self.layers[0].a.T) / m
        self.layers[1].db = np.sum(self.layers[1].dz) / m
        self.layers[0].dz = np.matmul(self.layers[1].w.T, self.layers[1].dz) * sigmoid_derivative(self.layers[0].a)
        self.layers[0].dw = np.matmul(self.layers[0].dz, x) / m
        self.layers[0].db = np.sum(self.layers[0].dz) / m
        for l in self.layers:
            l.w -= l.dw * self.learning_rate
            l.b -= l.db * self.learning_rate

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)

    @staticmethod
    def cost(y, t):
        loss = - t * np.log(y) - (1 - t) * np.log(1 - y)
        return np.sum(loss) / t.shape[0]
