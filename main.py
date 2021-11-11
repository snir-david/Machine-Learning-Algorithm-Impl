# (C) Snir David Nahari - 205686538
import collections
import sys

import numpy
import numpy as np


def getErrorRate(trained_xy, train_y):
    error = 0
    for i in range(len(train_y)):
        if trained_xy[i][1] != train_y[i]:
            error += 1
    return error


def knn(train_x, train_y):
    def takeSecond(elem):
        return elem[1]

    def classify_x(train_x, train_y, index, k):
        neighbors = []
        find_class = []
        class_of_x = -1
        max_count = 0
        for j in range(len(train_x)):
            if j != index:
                neighbors.append((train_x[j], np.linalg.norm(train_x[j] - train_x[index]), train_y[j]))
        neighbors.sort(key=takeSecond)
        for i in range(k):
            find_class.append(neighbors[i][2])
        for class_y in find_class:
            counter = find_class.count(class_y)
            if counter > max_count:
                class_of_x = class_y
        return class_of_x

    min_error = np.inf
    best_k = 0
    for k in range(1, 10):
        trained_xy = []
        for i in range(len(train_x)):
            trained_xy.append((train_x[i], classify_x(train_x, train_y, i, k)))
        error = getErrorRate(trained_xy, train_y)
        if error < min_error:
            min_error = error
            best_k = k
    return best_k


def perceptron(train_x, train_y):
    def find_weights_bias(train_x, train_y):
        classes = len(collections.Counter(train_y).keys())
        w = []
        for i in range(classes):
            w.append(np.zeros(len(train_x[0])))
        b = 0
        # TODO find best epoch number
        # TODO find learning rate
        # TODO find bias
        for epoch in range(250):
            for i in range(len(train_x)):
                # getting max arg from weights
                y_hat = np.argmax(np.sum(w * train_x[i], axis=1))
                if train_y[i] != y_hat:
                    w[int(train_y[i])] = w[int(train_y[i])] + train_x[i]
                    w[y_hat] = w[y_hat] - train_x[i]
        return w

    w = find_weights_bias(train_x, train_y)
    trained_xy = []
    for i in range(len(train_x)):
        trained_xy.append((train_x[i], np.argmax(np.sum(w * train_x[i], axis=1))))
    err = getErrorRate(trained_xy, train_y)
    return err


def svm(train_x, train_y):
    def find_weights_bias(train_x, train_y):
        classes = len(collections.Counter(train_y).keys())
        w = []
        for i in range(classes):
            w.append(np.zeros(len(train_x[0])))
        b = 0
        # TODO find best epoch number
        # TODO find learning rate
        # TODO find bias
        for epoch in range(250):
            for i in range(len(train_x)):
                # getting max arg from weights
                y_hat = np.argmax(np.sum(w * train_x[i], axis=1))
                if train_y[i] != y_hat:
                    w[int(train_y[i])] = w[int(train_y[i])] + train_x[i]
                    w[y_hat] = w[y_hat] - train_x[i]
        return w

    w = find_weights_bias(train_x, train_y)
    trained_xy = []
    for i in range(len(train_x)):
        trained_xy.append((train_x[i], np.argmax(np.sum(w * train_x[i], axis=1))))
    err = getErrorRate(trained_xy, train_y)
    return err

def passive_aggressive(train_x, train_y):
    def find_weights_bias(train_x, train_y):
        classes = len(collections.Counter(train_y).keys())
        w = []
        for i in range(classes):
            w.append(np.zeros(len(train_x[0])))
        b = 0
        # TODO find best epoch number
        # TODO find learning rate
        # TODO find bias
        for epoch in range(250):
            for i in range(len(train_x)):
                # getting max arg from weights
                y_hat = np.argmax(np.sum(w * train_x[i], axis=1))
                if train_y[i] != y_hat:
                    w[int(train_y[i])] = w[int(train_y[i])] + train_x[i]
                    w[y_hat] = w[y_hat] - train_x[i]
        return w

    w = find_weights_bias(train_x, train_y)
    trained_xy = []
    for i in range(len(train_x)):
        trained_xy.append((train_x[i], np.argmax(np.sum(w * train_x[i], axis=1))))
    err = getErrorRate(trained_xy, train_y)
    return err


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_x, train_y, test_s, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x = np.loadtxt(train_x, delimiter=",")
    train_y = np.loadtxt(train_y, delimiter=",")
    # knn_res = knn(train_x, train_y)
    # perc_res = perceptron(train_x, train_y)

