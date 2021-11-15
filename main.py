# (C) Snir David Nahari - 205686538
import collections
import sys
import numpy as np


def normalize_data(train_x):
    norm = np.linalg.norm(train_x, axis=0)
    normal_x = train_x / norm
    return normal_x


def shuffle_data(train_x, train_y):
    shuffeler = np.random.permutation(len(train_x))
    shuffled_x = train_x[shuffeler]
    shuffled_y = train_y[shuffeler]
    return shuffled_x, shuffled_y


def k_cross_fold(train_x, train_y, k):
    split_x = np.split(train_x, k)
    split_y = np.split(train_y, k)
    return split_x, split_y


def k_cross_join(train_x, train_y):
    # concatenate the training data
    tmp_arr_x = None
    tmp_arr_y = None
    for j in range(len(train_x)):
        if j == 0:
            tmp_arr_x = train_x[j]
            tmp_arr_y = train_y[j]
        else:
            tmp_arr_x = np.concatenate((tmp_arr_x, train_x[j]), axis=0)
            tmp_arr_y = np.concatenate((tmp_arr_y, train_y[j]), axis=0)
    return tmp_arr_x, tmp_arr_y


def getErrorRate(trained_xy, train_y):
    error = 0
    err_id = []
    for i in range(len(train_y)):
        if trained_xy[i][1] != train_y[i]:
            error += 1
            err_id.append(i)
    err_rate = error / len(train_y)
    return err_rate, err_id


def knn(train_x, train_y, test_x, k):
    def takeSecond(elem):
        return elem[1]

    def classify_x(train_x, train_y, index, classficate_x, k):
        neighbors = []
        find_class = []
        class_of_x = -1
        max_count = 0
        for j in range(len(train_x)):
            if j != index:
                neighbors.append((train_x[j], np.linalg.norm(train_x[j] - classficate_x), train_y[j]))
        neighbors.sort(key=takeSecond)
        for i in range(k):
            find_class.append(neighbors[i][2])
        for class_y in find_class:
            counter = find_class.count(class_y)
            if counter > max_count:
                class_of_x = class_y
        return class_of_x

    def find_k(train_x, train_y):
        min_err = np.inf  # initialize current error as infinity
        min_err_id = []
        # iterating from 1 to 100 try to find best k for current splitting
        sqrt_n = int(len(train_x) ** 0.5)
        for k in range(1, sqrt_n):
            trained_xy = []
            for i in range(len(train_x)):
                trained_xy.append((train_x[i], classify_x(train_x, train_y, i, train_x[i], k)))
            error, err_id = getErrorRate(trained_xy, train_y)
            if error < min_err:
                min_err = error
                min_err_id = err_id
                best_k = k
        print(f'best k is: {best_k},and the error is: {min_err} and Ids {min_err_id}')
        return best_k

    # predicts
    predictions = []
    for x in test_x:
        predictions.append(classify_x(train_x, train_y, -1, x, 1))
    return predictions


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
                    w[int(train_y[i])] += train_x[i] * 0.1
                    w[y_hat] -= train_x[i] * 0.1
        return w

    w = find_weights_bias(train_x, train_y)
    trained_xy = []
    for i in range(len(train_x)):
        trained_xy.append((train_x[i], np.argmax(np.sum(w * train_x[i], axis=1))))
    err = getErrorRate(trained_xy, train_y)
    print(err)
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
                    w[int(train_y[i])] = (1 - 0.6 * 0.2) * w[int(train_y[i])] + 0.6 * train_x[i]
                    w[y_hat] = w[y_hat] - train_x[i] * 0.6
                    for s in range(3):
                        if s != y_hat and s != train_y[i]:
                            w[s] = w[s] * (1 - 0.6 * 0.2)
                else:
                    for j in range(3):
                        w[j] = w[j] * (1 - 0.6 * 0.2)

        return w

    w = find_weights_bias(train_x, train_y)
    trained_xy = []
    for i in range(len(train_x)):
        trained_xy.append((train_x[i], np.argmax(np.sum(w * train_x[i], axis=1))))
    err = getErrorRate(trained_xy, train_y)
    print(err)
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
                # if train_y[i] != y_hat:
                if w[y_hat] * train_y[i] * train_x[i] < 0:
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
    test = np.loadtxt(test_s, delimiter=",")
    normal_x = normalize_data(train_x)
    pred_knn = knn(train_x, train_y, test, 1)
    # perc_res = perceptron(train_x, train_y)
    # svm_res = svm(train_x, train_y)
