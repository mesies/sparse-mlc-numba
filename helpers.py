rom
sklearn.datasets
import load_svmlight_file
from scipy.sparse import csr_matrix
# import numpy as np
import math
import time

import autograd.numpy as np
import autograd
# import numpy as np
import math
import time

import autograd
import autograd.numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

"""
This file contains helper functions
"""


def load_mlc_dataset(
        filename,
        header=True):
    """
    This function extends 'load_svmlight_file' so that datasets that have a header
    are parsed correctly

    Args:
        filename: Path of the dataset.
        header: True if file has a header.

    Returns:
        X_train: csr_matrix(TODO check) which contains the features of each point.
        y_train: csr_matrix(TODO check) which contains the labels of each point.
        header_info: False if there is no header, contains an array which contains
                    0: Number of training examples
                    1: Feature Dimensionality
                    2: Label Dimensionality
    """
    f = open(filename)

    header_info = False
    if header:
        header_info = f.readline().split()
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,

        )
        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        # Convert y to sparse array, Note : MultiLabelBinarizer() could be used
        ult = np.zeros((DATASET_SIZE, LABEL_NUMBER))
        for i in range(0, DATASET_SIZE):
            temp = np.zeros(LABEL_NUMBER)
            temp[np.asarray(y[i], dtype=int)] = 1
            ult[i] = temp
        y = csr_matrix(ult)

    else:
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,
        )
    f.close()

    return X, y, header_info


def sigmoid(x):
    if np.sign(x) >= 0:
        return 1. / (1. + math.exp(-x))
    else:
        return math.exp(x) / (1. + math.exp(x))


def log_likelihood(X, W, y):
    sigm = np.vectorize(sigmoid)
    # X.astype(np.double)
    # W.astype(np.double)
    # y.astype(np.double)
    xw = np.dot(X, W)
    sigm_xw = sigmA(xw)

    t = y
    y = sigm_xw

    loss = - np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
    # loss =  np.sum(y * np.log(sigm_wt_x) + (1 - y) * np.log(1 - sigm_wt_x))
    # loss = np.sum(t*xw - np.log(1 + np.exp(xw)))
    return loss


def sigmA(x):
    return 0.5 * (np.tanh(x) + 1)


def autotest(W, X, y):
    sigm = np.vectorize(sigmoid)
    # X.astype(np.double)
    # W.astype(np.double)
    # y.astype(np.double)


    loss = - np.sum(y * np.log(sigmA(np.dot(X, W))) + (1 - y) * np.log(1 - sigmA(np.dot(X, W))))
    # loss =  np.sum(y * np.log(sigm_wt_x) + (1 - y) * np.log(1 - sigm_wt_x))
    # loss = np.sum(t*xw - np.log(1 + np.exp(xw)))
    return loss


def log_likelihood_precalc(sigm_xw, y):
    return - np.sum(y * np.log(sigm_xw) + (1 - y) * np.log(1 - sigm_xw))


def gradient(X, W, y):
    # old_grad = - (X.T.dot(y)) + (X.T.dot(X)).dot(W)
    sigm = np.vectorize(sigmoid)
    X.astype(np.double)
    W.astype(np.double)
    y.astype(np.double)
    # xw = X.dot(W)
    xw = np.dot(X, W)
    sigm_xw = sigm(xw)

    neo = np.sum((sigm_xw.T - y) * X.T, axis=1)
    gradd = autograd.grad(log_likelihood, argnum=1)

    true_gradient = neo
    num_grad = gradd(X, W, y)

    print("Autograd and Analytic diff")
    print(np.linalg.norm(true_gradient - num_grad) / np.linalg.norm(true_gradient + num_grad))
    print("####")
    # return gradd(X, W, y)
    return neo


def gradient_precalc(X, sigm_xw, y):
    return np.sum((sigm_xw.T - y) * X.T, axis=1)


def grad_check(X, W, y):
    # Works

    K = W.shape[0]

    true_gradient = gradient(X, W, y)

    epsilon = 1e-6
    num_grad = np.zeros((K))

    for k in np.arange(0, K):
        W_tmp = W

        W_tmp[k] = W_tmp[k] + epsilon
        Ewplus = log_likelihood(X, W_tmp, y)

        W_tmp = W

        W_tmp[k] = W_tmp[k] - epsilon
        Ewminus = log_likelihood(X, W_tmp, y)

        num_grad[k] = np.divide((Ewplus - Ewminus), np.multiply(2, epsilon))

    true_gradient.reshape((3, 1))
    num_grad.reshape((3, 1))

    absmax = np.linalg.norm(true_gradient - num_grad) / np.linalg.norm(true_gradient + num_grad)
    if absmax > 0.5 * 1e-2:
        print("Warning : Failed Gradient Check " + str(absmax))
        exit(0)


def size(x):
    print(x.shape)


def tick():
    return time.time()


def toc(start_time, str=" "):
    print(str + " " + "--- %s seconds ---" % (time.time() - start_time))
