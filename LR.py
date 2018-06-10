import numpy as np
import pandas as pd

from activation import *

np.set_printoptions(threshold=np.inf)

path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', ',')
attributes = ['petal_width', 'petal_height', 'sepal_width', 'sepal_height', 'type']
data_set.columns = attributes
data_set.loc[data_set['type'] == 'Iris-setosa', 'type'] = 1
data_set.loc[data_set['type'] == 'Iris-virginica', 'type'] = 2
data_set.loc[data_set['type'] == 'Iris-versicolor', 'type'] = 3
X = data_set.drop('type', 1).values
Y = data_set['type'].values
Y = Y.reshape((Y.shape[0], 1))
file = 'iris.npz'
a = np.array([[0, 1, 0]])
for x in Y:
    if x == 1:
        a = np.append(a, np.array([[0, 0, 1]], dtype=np.float64), axis=0)
    elif x == 2:
        a = np.append(a, np.array([[0, 1, 0]], dtype=np.float64), axis=0)
    elif x == 3:
        a = np.append(a, np.array([[1, 0, 0]], dtype=np.float64), axis=0)
Y = np.delete(a, 0, axis=0).T


class LR(object):
    def __init__(self, inputs=2, outputs=1, readFile=None):
        if readFile is None:
            self.w = np.random.randn(outputs, inputs)
            self.b = np.zeros([outputs, 1])
        else:
            self.w = np.load(file)['w']
            self.b = np.load(file)['b']

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
        return - np.multiply(t, np.log(y)).sum()


n = LR(4, 3, file)


def lr_test(x):
    o = n.forward(x)
    print("%.5f %.5f %.5f" % (o[0][0], o[0][1], o[0][2]))


def lr_train():
    for i in range(20000):
        n.train(X, Y)
        print(n.cost(n.forward(X), Y))


if __name__ == '__main__':
    lr_train()
    # lr_test(X[30])
    np.savez(file, w=n.w, b=n.b)
