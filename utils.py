import numpy as np

__all__ = [
    'Layer'
]


class Layer(object):
    def __init__(self, inputs, classes):
        self.w = np.random.randn(classes, inputs)
        self.b = np.zeros([classes, 1])
        self.a = None  # This is the output of the previous Layer
        self.z = None
        self.dz = None
        self.dw = None
        self.db = None

    def back_propagation(self, da, learning_rate):
        m = self.a.shape[0]
        self.dz = da * self.activation_derivative(self.z)
        self.dw = np.matmul(self.dz, self.a.T) / m
        self.db = np.sum(self.dz, axis=1, keepdims=True) / m
        a = np.multiply(self.w.T, self.dz)
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate
        return a

    def forward_propagation(self, a):
        self.a = a
        self.z = np.matmul(self.w, self.a) + self.b
        return self.activation()

    def activation_derivative(self, z):
        pass

    def activation(self):
        pass