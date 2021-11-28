# (C) Snir David Nahari - 205686538
import sys
# from datetime import datetime

import numpy as np


# normalize data for better results
# dividing the x-values by it's norms
def normalize_data(train_x):
    # normalize data using norm for each feature
    norm = np.linalg.norm(train_x, axis=0)
    normal_x = train_x / norm
    return normal_x


def z_score(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normal_data = (data - mean) / std
    return normal_data


# shuffling data randomly
def shuffle_data(train_x, train_y):
    shuffeler = np.random.permutation(len(train_x))
    shuffled_x = train_x[shuffeler]
    shuffled_y = train_y[shuffeler]
    return shuffled_x, shuffled_y


# getting arguments that maximize the phrase - weights[i]*train[i]
# optional - getting argument r that won't including in arg max, default value of r is -1
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


# hinge function loss - maximum between (0, 1- w_yx + w_rx)
def hinge_loss(w, y, x, arg_max):
    wyx = np.sum(w[y] * x)
    wrx = np.sum(w[arg_max] * x)
    return np.maximum(0, 1 - wyx + wrx)


# calculating the error rate - how much labels are not identical to train y
def get_error_rate(trained_xy, train_y):
    error = 0
    for i in range(len(train_y)):
        if trained_xy[i][1] != train_y[i]:
            error += 1
    err_rate = error / len(train_y)
    return err_rate


# KNN algorithm - getting train data, finding best k
# on test data - checking with k found
def knn(train_x, train_y, test_x, k):
    # helper method for sorting according to norm
    def takeSecond(elem):
        return elem[1]

    # knn method that classifies points
    def classify_x(train_x, train_y, classifies_x, k, index=-1):
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
        return int(class_of_x)

    # this function was used to find best k for training set
    def training(train_x, train_y):
        best_k = 0
        min_err = np.inf  # initialize current error as infinity
        # iterating from 1 to 100 try to find best k for current splitting
        sqrt_n = int(len(train_x) ** 0.5)
        for k in range(1, sqrt_n):
            trained_xy = []
            for i in range(len(train_x)):
                trained_xy.append((train_x[i],
                                   classify_x(train_x=train_x, train_y=train_y, index=i, classifies_x=train_x[i],
                                              k=round(k))))
            error = get_error_rate(trained_xy, train_y)
            if error < min_err:
                min_err = error
                best_k = k
        return best_k

    best_k = training(train_x, train_y)
    # predicts new data labels
    predictions = []
    for x in test_x:
        predictions.append(classify_x(train_x, train_y, x, best_k))
    return predictions


# Perceptron algorithm - getting train data, finding weights
# on test data - checking with weights found
def perceptron(train_x, train_y, test_x):
    # return arg max on x with weights
    def classify_x(classifies_x, weights):
        sum = np.sum(weights * classifies_x, axis=1)
        return np.argmax(sum)

    def find_weights_bias(train_x, train_y, learning_rate, epochs):
        # initialize ndarray 2-D , each class has array with features length
        w_perc = np.array([np.zeros(len(train_x[0])), np.zeros(len(train_x[0])), np.zeros(len(train_x[0]))])
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
                    # checking for error
                    for i in range(len(train_x)):
                        trained_xy.append((train_x[i], classify_x(train_x[i], w_perc)))
                    err = get_error_rate(trained_xy, train_y)
                    if err < min_err_perc:
                        min_err_perc = err
                        best_epoch_num = epoch
                        best_weight = w_perc
        return best_weight, min_err_perc, best_epoch_num

    def training(train_x, train_y):
        # output = open("perceptron_parma.txt", 'w+')
        # start = datetime.now()
        # st_current_time = start.strftime("%H:%M:%S")
        # output.write(f'Start training at -  {st_current_time}\n')
        w, min_err, ep = find_weights_bias(train_x, train_y, 1, 100)
        # output.write(f'minimum error: {min_err} in epoch: {ep} and weights are: {w}\n')
        # end = datetime.now()
        # end_current_time = end.strftime("%H:%M:%S")
        # output.write(f'End training at - {end_current_time}\n')
        # run = end - start
        # output.write(f'Total time for training is {run}\n\n')
        return w

    best_weights_found = training(train_x, train_y)
    # predicts
    # best_weights_found = np.array([([-0.10061318, -4.39267569, 22.77108818, 12.26794959, -1.37759287, -9.]),
    #                                ([6.93356053, -0.6889391, -5.57647865, -2.04780874, 1.52745653, 15.]),
    #                                ([-6.83294735, 5.08161479, -17.19460953, -10.22014085, -0.14986366, -6.])])
    predictions = []
    for x in test_x:
        predictions.append(classify_x(x, best_weights_found))
    return predictions


# SVM algorithm - getting train data, finding weights
# on test data - checking with weights found
def svm(train_x, train_y, test_x):
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
                    trained_xy.append((train_x[i], arg_max(w_svm, train_x[i])))
                err = get_error_rate(trained_xy, train_y)
                # print(f'error rate is: {err}, in epoch: {epoch}')
                if err < min_err_svm:
                    min_err_svm = err
                    best_epoch_num = epoch
                    best_weight = w_svm
            # eta_svm = eta_svm / 2
        return best_weight, min_err_svm, best_epoch_num

    def training(train_x, train_y):
        # output = open("svm_parma.txt", 'w+')
        # values = [1, 0.8, 0.7, 0.5, 0.3, 0.1, 0.001, 0.0001, 0]
        # min_err_arr = []
        # best_w = []
        # for i in range(len(values)):
        #     output.write(f'Starting with new learning rate...\n')
        #     for j in range(len(values)):
        #         start = datetime.now()
        #         st_current_time = start.strftime("%H:%M:%S")
        #         output.write(f'Start training at -  {st_current_time}\n')
        w, min_err, ep = find_weights_bias(train_x, train_y, 0.1, 0.1, 50)
        #         min_err_arr.append(min_err)
        #         best_w.append(w)
        #         output.write(f'minimum error: {min_err} in epoch: {ep} and weights are: {w} \n'
        #                      f'with regularization: {values[j]} and learning rate: {values[i]}\n')
        #         end = datetime.now()
        #         end_current_time = end.strftime("%H:%M:%S")
        #         output.write(f'End training at - {end_current_time}\n')
        #         run = end - start
        #         output.write(f'Total time for training is {run}\n\n')
        # min_e = min(min_err_arr)
        # output.write(
        #     f'Minimum Error in all training is {min_e} and the weights are {best_w[min_err_arr.index(min_e)]}\n')
        # output.close()
        return w

    best_weights_found = training(train_x, train_y)
    # predicts
    # best_weights_found = np.array([([0.18710489, 0.00353916, 0.67677144, 0.69887273, 0.05898188, -0.34122238]),
    #                                ([0.10580316, -0.45370831, 0.02953652, -0.15403748, -0.00902337, 0.31851379]),
    #                                ([-0.29290805, 0.45016915, -0.70630796, -0.54483525, -0.04995851, 0.02270859])])
    predictions = []
    for x in test_x:
        predictions.append(arg_max(best_weights_found, x))
    return predictions


# Passive aggressive algorithm - getting train data, finding weights
# on test data - checking with weights found
def passive_aggressive(train_x, train_y, test_x):
    def find_weights_bias(train_x, train_y, epochs):
        # initialize weights with arrays with num of features
        w_pa = np.array([np.zeros(len(train_x[0])), np.zeros(len(train_x[0])), np.zeros(len(train_x[0]))])
        min_err_svm = np.inf
        best_epoch_num = 0
        best_weight = []
        for epoch in range(epochs):
            # train_x, train_y = shuffle_data(train_x, train_y)
            for i in range(len(train_x)):
                y_hat = arg_max(w_pa, train_x[i], train_y[i])
                loss = hinge_loss(w_pa, int(train_y[i]), train_x[i], y_hat)
                if loss > 0:
                    tau = (loss / (2 * ((np.linalg.norm(train_x[i])) ** 2)))
                    w_pa[int(train_y[i])] += train_x[i] * tau
                    w_pa[y_hat] -= train_x[i] * tau
                    # check error rate with current parameters
                    trained_xy = []
                    for i in range(len(train_x)):
                        trained_xy.append((train_x[i], arg_max(w_pa, train_x[i])))
                    err = get_error_rate(trained_xy, train_y)
                    if err < min_err_svm:
                        min_err_svm = err
                        best_epoch_num = epoch
                        best_weight = w_pa
        return best_weight, min_err_svm, best_epoch_num

    def training(train_x, train_y):
        # output = open("pa_parma.txt", 'w+')
        # min_err_arr = []
        # best_w = []
        epoch = 20
        # output.write(f'Starting new iteration...\n')
        # start = datetime.now()
        # st_current_time = start.strftime("%H:%M:%S")
        # output.write(f'Start training at -  {st_current_time}\n')
        w, min_err, ep = find_weights_bias(train_x, train_y, epoch)
        # min_err_arr.append(min_err)
        # best_w.append(w)
        # output.write(f'minimum error: {min_err} in epoch: {ep} and weights are: {w} \n')
        # end = datetime.now()
        # end_current_time = end.strftime("%H:%M:%S")
        # output.write(f'End training at - {end_current_time}\n')
        # run = end - start
        # output.write(f'Total time for training is {run}\n\n')
        # min_e = min(min_err_arr)
        # output.write(
        #     f'Minimum Error in all training is {min_e} and the weights are {best_w[min_err_arr.index(min_e)]}\n')
        # output.close()
        return w

    best_weights_found = training(train_x, train_y)
    # predicts
    # best_weights_found = np.array([([0.61320707, -1.14797863, 4.24116836, 3.43345916, -0.23288381, -2.98588669]),
    #                                ([0.31033628, 0.25195826, -1.34558459, -1.09893239, 0.41667218, 2.87038145]),
    #                                ([-0.92354335, 0.89602037, -2.89558377, -2.33452677, - 0.18378837, 0.11550524])])
    predictions = []
    for x in test_x:
        predictions.append(arg_max(best_weights_found, x))
    return predictions


if __name__ == '__main__':
    # start = datetime.now()
    # st_current_time = start.strftime("%H:%M:%S")
    # print(f'Start training at -  {st_current_time}\n')
    train_x, train_y, test_s, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # loading data
    train_x = np.loadtxt(train_x, delimiter=",")
    train_y = np.loadtxt(train_y, delimiter=",")
    test_s = np.loadtxt(test_s, delimiter=",")
    # normalizing and shuffle data
    normal_x = z_score(train_x)
    normal_test = z_score(test_s)
    shuffled_x, shuffled_y = shuffle_data(normal_x, train_y)
    # adding bias for train and test
    shuffled_x_1 = np.c_[shuffled_x, np.ones(len(shuffled_x))]
    normal_x_1 = np.c_[normal_test, np.ones(len(normal_test))]

    # predict test data
    pred_knn = knn(train_x, train_y, test_s, 1)
    pred_prec = perceptron(shuffled_x_1, shuffled_y, normal_x_1)
    pred_svm = svm(shuffled_x_1, shuffled_y, normal_x_1)
    pred_pa = passive_aggressive(shuffled_x_1, shuffled_y, normal_x_1)
    # printing data
    out = open(output_file, '+w')
    for i in range(len(normal_test)):
        out.write(f"knn: {pred_knn[i]}, perceptron: {pred_prec[i]}, svm: {pred_svm[i]}, pa: {pred_pa[i]}\n")
    out.close()
    # end = datetime.now()
    # end_current_time = end.strftime("%H:%M:%S")
    # print(f'End training at - {end_current_time}\n')
