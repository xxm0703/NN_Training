import numpy as np
import pandas as pd

from activation import sigmoid

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
tmp = np.array([[0, 1, 0]])
for x in Y:
    if x == 1:
        tmp = np.append(tmp, np.array([[0, 0, 1]]), axis=0)
    elif x == 2:
        tmp = np.append(tmp, np.array([[0, 1, 0]]), axis=0)
    elif x == 3:
        tmp = np.append(tmp, np.array([[1, 0, 0]]), axis=0)
Y = np.delete(tmp, 0, axis=0).T


class LR(object):
    def __init__(self, inputs=2, outputs=1, file_name=None):
        if file is None:
            self.w = np.random.randn(outputs, inputs)
            self.b = np.zeros([outputs, 1])
        else:
            self.w = np.load(file_name)['w']
            self.b = np.load(file_name)['b']

        self.learning_rate = 0.18
        self.dz = 0

    def forward(self, input_data):
        output = np.matmul(self.w, input_data.T) + self.b
        output = sigmoid(output)
        return output

    def backward(self, input_data, y, o):
        m = input_data.shape[0]
        self.dz = o - y
        dw = np.matmul(self.dz, input_data) / m
        db = np.sum(self.dz) / m
        a = np.multiply(self.learning_rate, dw)
        self.w -= a
        self.b -= db * self.learning_rate

    def train(self, test_input, y):
        o = self.forward(test_input)
        self.backward(test_input, y, o)

    @staticmethod
    def cost(y, t):
        return - np.multiply(t, np.log(y)).sum()


n = LR(4, 3, file)


def lr_test(test_set):
    o = n.forward(test_set)
    print("%.5f %.5f %.5f" % (o[0][0], o[0][1], o[0][2]))


def lr_train():
    for i in range(20000):
        n.train(X, Y)
        print(n.cost(n.forward(X), Y))


if __name__ == '__main__':
    # lr_train()
    lr_test(X[30])
    np.savez(file, w=n.w, b=n.b)
