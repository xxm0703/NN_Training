import numpy as np
from activation import sigmoid

__all__ = ['LR']


class LR(object):
    def __init__(self, inputs=2, outputs=1, read_file=None):
        if read_file is None:
            self.w = np.random.randn(outputs, inputs)
            self.b = np.zeros([outputs, 1])
        else:
            self.w = np.load(read_file)['w']
            self.b = np.load(read_file)['b']

        self.learning_rate = 0.18
        self.dz = 0

    def forward(self, x):
        output = np.matmul(self.w, x.T) + self.b
        output = sigmoid(output)
        return output

    def backward(self, x, y, o):
        m = x.shape[0]
        self.dz = o - y
        dw = np.matmul(self.dz, x) / m
        db = np.sum(self.dz) / m
        a = np.multiply(self.learning_rate, dw)
        self.w -= a
        self.b -= db * self.learning_rate

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)

    @staticmethod
    def cost(y, t):
        loss = - t * np.log(y) - (1 - t) * np.log(1 - y)
        return np.sum(loss) / t.shape[0]
