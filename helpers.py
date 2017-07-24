
import autograd
import numpy as npp
import math
import time
import numpy as np
# import autograd.numpy as np
import scipy
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from scipy.special import logsumexp

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
        :param filename: Path of the dataset.
        :param header: True if file has a header.

    Returns:
        :returns X_train: csr_matrix(TODO check) which contains the features of each point.
        :returns y_train: csr_matrix(TODO check) which contains the labels of each point.
        :returns header_info: False if there is no header, contains an array which contains
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
    """
    An Implementation of the sigmoid function
    """
    # return 0.5 * (np.tanh(0.5*x) + 1)
    # return scipy.special.expit(x)
    return 1. / (1. + np.exp(-x))


def log_likelihood(X, W, y):
    """
    L = - np.sum(t * np.log(sigmoid(np.dot(X, W))) + (1 - t) * np.log(1 - sigmoid(np.dot(X, W))))
    which can be written as
    xw_hat = ((-1)**(1-t)) * np.dot(X, W)
    L = -np.sum(-np.log(1 + np.exp(-xw_hat)))


    L = - sum(Yn)

    Yn = log(sigmoid(X*W)) if t = 1
    Yn = log(1 - sigmoid(X*W) if t = 0
    =>
    Yn = log(sigmoid(X*W)) if t = 1
    Yn = log(1 - sigmoid(X*W) if t = 0
    ^
    1 - sigmoid(x) = sigmoid(-x)
    =>
    Yn = log(sigmoid(X*W)) if t = 1
    Yn = log(sigmoid((-1)*X*W) if t = 0
    =>
    Yn = log(sigmoid((-1)^(t-1) * (X*W))
    ^
    log(sigmoid(x)) = log(1 / 1 + exp(-x)) = -log(1 + exp(-x))
    =>
    L = -np.sum(-np.log(1 + np.exp(-((-1)**(1-t)) * np.dot(X, W))))

    This functions returns the log likelihood that will be maximized.
    :param X: Training examples
    :param W: Weight vector
    :param y: True Categories of the training examples X
    :return: minus log likelihood
    """

    t = y

    xw_hat = ((-1)**(1-t)) * np.dot(X, W)
    L = -np.sum(-np.log(1 + np.exp(-xw_hat)))

    return L


def logsumexp(seq):
    size(seq)
    maxx = np.max(seq)
    summ = np.exp(seq-maxx).sum()
    return maxx + np.log(1 + summ)



def auto_gradient(X, W, y):
    """
    Gradient of log_likelihood calculated with use of autograd package
    :param X: Training examples
    :param W: Weight vector
    :param y: True Categories of the training examples X
    :return: Gradient
    """
    gradient = autograd.grad(log_likelihood, argnum=1)
    return gradient(X, W, y)


def gradient(X, W, y):
    """
    Gradient of log_likelihood
    :param X: Training examples
    :param W: Weight vector
    :param y: True Categories of the training examples X
    :return: Gradient
    """
    # old_grad = - (X.T.dot(y)) + (X.T.dot(X)).dot(W)

    sigm_xw = sigmoid(np.dot(X, W))

    neo = np.sum((sigm_xw.T - y) * X.T, axis=1)

    return neo


def gradient_precalc(X, sigm_xw, y):
    """
        Gradient of log_likelihood that makes use of precomputed data.
        :param X: Training examples
        :param W: Weight vector
        :param y: True Categories of the training examples X
        :return: Gradient
        """
    return np.sum((sigm_xw.T - y) * X.T, axis=1)


def grad_check(X, W, y):
    """
    Checks the validation of gradient versus a numerical approximation
    Note : Even though gradient and auto_auto_gradient return both the same values, meaning gradient is correct,
    grad_check's numerical gradient is 0.5 * gradient
    :param X: Training examples
    :param W: Weight vector
    :param y: True Categories of the training examples X
    """
    # Works but the numeric gradient is double than the actual gradient for some reason

    true_gradient = gradient(X, W, y)

    epsilon = 1e-4
    num_grad = np.zeros(W.shape[0])

    for k in np.arange(0, W.shape[0]):
        W_tmp = W

        W_tmp[k] = W_tmp[k] + epsilon
        Ewplus = log_likelihood(X, W_tmp, y)

        W_tmp = W

        W_tmp[k] = W_tmp[k] - epsilon
        Ewminus = log_likelihood(X, W_tmp, y)


        num_grad[k] = np.divide((np.subtract(Ewplus, Ewminus)), np.multiply(1, epsilon))

    true_gradient.reshape((W.shape[0], 1))
    num_grad.reshape((W.shape[0], 1))

    absmax = np.linalg.norm(true_gradient - num_grad) / np.linalg.norm(true_gradient + num_grad)
    if absmax > 0.5 * 1e-2:
        return 'WARNING: Failed Gradient Check ' + str(absmax)
    return 'INFO: Successful Gradient Check ' + str(absmax)


def plot_linreg_results(X, W, y, preds, lossHistory):


    print("Score " + str(accuracy_score(y_true=y, y_pred=preds)))

    Y = (-W[0] - (W[1] * X)) / W[2]

    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
    plt.plot(X, Y, "r-")

    fig = plt.figure()

    plt.plot(np.arange(0, len(lossHistory)), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


def size(x):
    """
    Prints shape of x.
    """
    print(x.shape)


def tic():
    """
    MATLAB's tic.
    """
    return time.time()


def toc(start_time, str=" "):
    """
    MATLAB's toc.
    :param start_time: Time supplied by tic
    :param str: Extra string to be included in the beginning of print statement
    """
    print(str + " " + "--- %s seconds ---" % (time.time() - start_time))
