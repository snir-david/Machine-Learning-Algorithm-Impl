# (C) Snir David Nahari - 205686538
import collections
import sys
from datetime import datetime
import numpy as np


def normalize_data(train_x):
    # normalize data using norm for each feature
    norm = np.linalg.norm(train_x, axis=0)
    normal_x = train_x / norm
    # mean = np.mean(train_x, axis=0)
    # std = np.std(train_x, axis=0)
    # normal_x = (train_x - mean) / std
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


def arg_max(weights, train_x, r=-1):
    max_id = -1
    max_v = -np.inf
    values = np.sum(weights * train_x, axis=1)
    for index in range(len(values)):
        if index != r:
            if max_v < values[index]:
                max_id = index
                max_v = values[index]
    return max_id


def hinge_loss(w, y, x, arg_max):
    wyx = np.sum(w[y] * x)
    wrx = np.sum(w[arg_max] * x)
    return np.maximum(0, 1 - wyx + wrx)


def get_error_rate(trained_xy, train_y):
    error = 0
    for i in range(len(train_y)):
        if trained_xy[i][1] != train_y[i]:
            error += 1
    err_rate = error / len(train_y)
    return err_rate


def knn(train_x, train_y, test_x, k):
    # helper method for sorting according to norm
    def takeSecond(elem):
        return elem[1]

    # knn method that classifies points
    def classify_x(train_x, train_y, index, classifies_x, k):
        neighbors = []
        find_class = []
        class_of_x = -1
        max_count = 0
        # adding all neighbors distance
        for j in range(len(train_x)):
            if j != index:
                neighbors.append((train_x[j], np.linalg.norm(train_x[j] - classifies_x), train_y[j]))
        # sorting list of neighbors according to distance
        neighbors.sort(key=takeSecond)
        # finding k closest neighbors
        for i in range(k):
            find_class.append(neighbors[i][2])
        # finding most common class
        for class_y in find_class:
            counter = find_class.count(class_y)
            if counter > max_count:
                class_of_x = class_y
        return class_of_x

    # this function was used to find best k for training set
    def find_k(train_x, train_y):
        min_err = np.inf  # initialize current error as infinity
        min_err_id = []
        # iterating from 1 to 100 try to find best k for current splitting
        sqrt_n = int(len(train_x) ** 0.5)
        for k in range(1, sqrt_n):
            trained_xy = []
            for i in range(len(train_x)):
                trained_xy.append((train_x[i], classify_x(train_x, train_y, i, train_x[i], k)))
            error, err_id = get_error_rate(trained_xy, train_y)
            if error < min_err:
                min_err = error
                min_err_id = err_id
                best_k = k
        # print(f'best k is: {best_k},and the error is: {min_err} and Ids {min_err_id}')
        return best_k

    # find_k(train_x, train_y)
    # predicts
    predictions = []
    for x in test_x:
        predictions.append(classify_x(train_x, train_y, -1, x, k))
    return predictions


def perceptron(train_x, train_y, test_x):
    def classify_x(classifies_x, weights):
        sum = np.sum(weights * classifies_x, axis=1)
        return np.argmax(sum)

    def find_weights_bias(train_x, train_y, learning_rate, epochs):
        classes = len(collections.Counter(train_y).keys())
        w_perc = []
        for i in range(classes):
            w_perc.append(np.zeros(len(train_x[0])))
        min_err_perc = np.inf
        best_epoch_num = 0
        best_weight = []
        for epoch in range(epochs):
            train_x, train_y = shuffle_data(train_x, train_y)
            for i in range(len(train_x)):
                # getting max arg from weights
                y_hat = np.argmax(np.sum(w_perc * train_x[i], axis=1))
                if int(train_y[i]) != y_hat:
                    w_perc[int(train_y[i])] += train_x[i] * learning_rate
                    w_perc[y_hat] -= train_x[i] * learning_rate
                    trained_xy = []
                    for i in range(len(train_x)):
                        trained_xy.append((train_x[i], classify_x(train_x[i], w_perc)))
                    err = get_error_rate(trained_xy, train_y)
                    # print(f'error rate is: {err}, in epoch: {epoch}')
                    if err < min_err_perc:
                        min_err_perc = err
                        best_epoch_num = epoch
                        best_weight = w_perc
        return best_weight, min_err_perc, best_epoch_num

    def training(train_x, train_y):
        start = datetime.now()
        st_current_time = start.strftime("%H:%M:%S")
        print("Start training at - ", st_current_time)
        w, min_err, ep = find_weights_bias(train_x, train_y, 1, 1000)
        print(f'minimum error: {min_err} in epoch: {ep} and weights are: {w}')
        end = datetime.now()
        end_current_time = end.strftime("%H:%M:%S")
        print("End training at - ", end_current_time)
        run = end - start
        print("Total time for training is ", run)

    # training(train_x, train_y)
    # predicts
    best_weights_found = np.array([([3.92604469, -1.30013308, 5.41772099, -2.96697031, 0.08660937]),
                                   ([-1.26011623, 0.28110985, -0.21050135, 0.62153066, -0.06218109]),
                                   ([-2.66592846, 1.01902322, -5.20721964, 2.34543964, -0.02442828])])
    predictions = []
    for x in test_x:
        predictions.append(classify_x(x, best_weights_found))
    return predictions


def svm(train_x, train_y, test_x):
    def classify_x(classifies_x, weights):
        return arg_max(classifies_x, weights)

    def find_weights_bias(train_x, train_y, eta_svm, lambda_svm, epochs):
        w_svm = np.array([np.zeros(len(train_x[0])), np.zeros(len(train_x[0])), np.zeros(len(train_x[0]))])
        min_err_svm = np.inf
        best_epoch_num = 0
        best_weight = []
        for epoch in range(epochs):
            # train_x, train_y = shuffle_data(train_x, train_y)
            for i in range(len(train_x)):
                r = arg_max(w_svm, train_x[i], train_y[i])
                if hinge_loss(w_svm, int(train_y[i]), train_x[i], r) > 0:
                    w_svm[int(train_y[i])] = (1 - lambda_svm * eta_svm) * w_svm[int(train_y[i])] + train_x[i] * eta_svm
                    w_svm[r] = (1 - lambda_svm * eta_svm) * w_svm[r] - train_x[i] * eta_svm
                    for w in range(len(w_svm)):
                        if not (np.array_equal(w_svm[w], w_svm[r]) or np.array_equal(w_svm[w], w_svm[int(train_y[i])])):
                            w_svm[w] = (1 - lambda_svm * eta_svm) * w_svm[w]
                else:
                    for w in range(len(w_svm)):
                        w_svm[w] *= (1 - lambda_svm * eta_svm)
                # check error rate with current parameters
                trained_xy = []
                for i in range(len(train_x)):
                    trained_xy.append((train_x[i], classify_x(train_x[i], w_svm)))
                err = get_error_rate(trained_xy, train_y)
                # print(f'error rate is: {err}, in epoch: {epoch}')
                if err < min_err_svm:
                    min_err_svm = err
                    best_epoch_num = epoch
                    best_weight = w_svm
            eta_svm = eta_svm / 2
        return best_weight, min_err_svm, best_epoch_num

    def training(train_x, train_y):
        output = open("svm_parma.txt", 'w+')
        values = [0.6, 0.4, 0.2, 0.1, 0.001, 0.0001]
        for i in range(len(values)):
            output.write(f'Starting with new learning rate...\n')
            for j in range(len(values)):
                start = datetime.now()
                st_current_time = start.strftime("%H:%M:%S")
                output.write(f'Start training at -  {st_current_time}\n')
                w, min_err, ep = find_weights_bias(train_x, train_y, values[j], values[i], 200)
                output.write(f'minimum error: {min_err} in epoch: {ep} and weights are: {w} \n'
                             f'with regularization: {values[j]} and learning rate: {values[i]}\n')
                end = datetime.now()
                end_current_time = end.strftime("%H:%M:%S")
                output.write(f'End training at - {end_current_time}\n')
                run = end - start
                output.write(f'Total time for training is {run}\n\n')

    training(train_x, train_y)
    # predicts
    # best_weights_found = np.array([([3.92604469, -1.30013308, 5.41772099, -2.96697031, 0.08660937]),
    #                                ([-1.26011623, 0.28110985, -0.21050135, 0.62153066, -0.06218109]),
    #                                ([-2.66592846, 1.01902322, -5.20721964, 2.34543964, -0.02442828])])
    # predictions = []
    # for x in test_x:
    #     predictions.append(classify_x(x, best_weights_found))
    # return predictions


def passive_aggressive(train_x, train_y, test_x):
    def classify_x(classifies_x, weights):
        return arg_max(classifies_x, weights)

    def find_weights_bias(train_x, train_y, epochs):
        w_pa = np.array([np.zeros(len(train_x[0])), np.zeros(len(train_x[0])), np.zeros(len(train_x[0]))])
        min_err_svm = np.inf
        best_epoch_num = 0
        best_weight = []
        for epoch in range(epochs):
            # train_x, train_y = shuffle_data(train_x, train_y)
            for i in range(len(train_x)):
                y_hat = arg_max(w_pa,train_x[i], train_y[i])
                loss = hinge_loss(w_pa, int(train_y[i]), train_x[i], y_hat)
                if loss > 0:
                    tau = (loss / 2 * (np.linalg.norm(train_x[i]) ** 2))
                    w_pa[int(train_y[i])] += train_x[i] * tau
                    w_pa[y_hat] -= train_x[i] * tau
                # check error rate with current parameters
                trained_xy = []
                for i in range(len(train_x)):
                    trained_xy.append((train_x[i], classify_x(train_x[i], w_pa)))
                err = get_error_rate(trained_xy, train_y)
                # print(f'error rate is: {err}, in epoch: {epoch}')
                if err < min_err_svm:
                    min_err_svm = err
                    best_epoch_num = epoch
                    best_weight = w_pa
        return best_weight, min_err_svm, best_epoch_num

    def training(train_x, train_y):
        output = open("pa_parma.txt", 'w+')
        for i in range(5):
            epoch = 100
            output.write(f'Starting new iteration...\n')
            start = datetime.now()
            st_current_time = start.strftime("%H:%M:%S")
            output.write(f'Start training at -  {st_current_time}\n')
            w, min_err, ep = find_weights_bias(train_x, train_y, epoch)
            output.write(f'minimum error: {min_err} in epoch: {ep} and weights are: {w} \n')
            end = datetime.now()
            end_current_time = end.strftime("%H:%M:%S")
            output.write(f'End training at - {end_current_time}\n')
            run = end - start
            output.write(f'Total time for training is {run}\n\n')
            epoch += 100

    training(train_x, train_y)
    # predicts
    # best_weights_found = np.array([([3.92604469, -1.30013308, 5.41772099, -2.96697031, 0.08660937]),
    #                                ([-1.26011623, 0.28110985, -0.21050135, 0.62153066, -0.06218109]),
    #                                ([-2.66592846, 1.01902322, -5.20721964, 2.34543964, -0.02442828])])
    # predictions = []
    # for x in test_x:
    #     predictions.append(classify_x(x, best_weights_found))
    # return predictions


if __name__ == '__main__':
    train_x, train_y, test_s, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x = np.loadtxt(train_x, delimiter=",")
    train_y = np.loadtxt(train_y, delimiter=",")
    test = np.loadtxt(test_s, delimiter=",")
    normal_x = normalize_data(train_x)
    shuffled_x, shuffled_y = shuffle_data(normal_x, train_y)
    normal_test = normalize_data(test)
    # pred_knn = knn(train_x, train_y, test, 1)
    # pred_prec = perceptron(shuffled_x, shuffled_y, normal_test)
    # svm_res = svm(shuffled_x, shuffled_y, normal_test)
    pred_pa = passive_aggressive(shuffled_x, shuffled_y, normal_test)
