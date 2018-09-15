import numpy as np
from activation import *


class NN(object):
    def __init__(self, inputs=2, classes=1, hidden_layers=1, nodes=(2,)):
        self.n_inputs = inputs
        self.n_classes = classes
        self.n_hidden = hidden_layers
        self.n_nodes = nodes
        self.output_layer = Layer(nodes[-1], classes)
        self.hidden_layers = [Layer(inputs, nodes[0])]
        for i in range(1, hidden_layers):
            self.hidden_layers.append(Layer(nodes[i - 1], nodes[i + 1]))

    def forward_prop(self, data):
        output = np.multiply(self.hidden_layers[0].weights, data.T)
        output = np.add(output.T, self.hidden_layers[0].bias)
        output = sigmoid(output)

        for l in self.hidden_layers[1:]:
            output = np.multiply(l, output.T)
            output = np.add(output.T, l.bias)
            output = sigmoid(output)

        output = np.matmul(self.output_layer.weights, np.transpose(output))
        output = np.add(np.transpose(output), self.output_layer.bias)
        output = sigmoid(output)

        return output

    def back_prop(self, data):
        self.output_layer.dz = data * sigmoid_derivative(self.output_layer.z)
        self.output_layer.dw = (self.output_layer.z * self.hidden_layers[-1].a.T) / self.m
        self.output_layer.db = np.sum(self.output_layer.z, axis=1, keepdims=True) / self.m
        self.hidden_layers[-1].da = self.output_layer.weights.T * self.output_layer.dz
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[i].dz = data * sigmoid_derivative(self.hidden_layers[i].z)
            self.hidden_layers[i].dw = self.hidden_layers[i].z * self.hidden_layers[i - 1].a
            self.hidden_layers[i].db = self.hidden_layers[i].z
            self.hidden_layers[i - 1].da = self.hidden_layers[i].weights.T * self.hidden_layers[i].dz
