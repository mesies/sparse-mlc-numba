# import autograd
# import autograd.numpy as np
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import cython
import tqdm
from mathutil import gradient, log_likelihood, gradient_sp, log_likelihood_sp

"""
This file contains helper functions
"""

def load_mlc_dataset(
        filename,
        header=True,
        concatbias=True):
    """
    This function extends 'load_svmlight_file' so that datasets that have a header
    are parsed correctly

    Args:
        :param filename: Path of the dataset.
        :param header: True if file has a header.
        :param concatbias: Whether to add bias.

    Returns:
        :returns X_train: csr_matrix(TODO check) which contains the features of each point.
        :returns y_train: csr_matrix(TODO check) which contains the labels of each point.
        :returns header_info: False if there is no header, contains an array which contains
                    0: Number of training examples
                    1: Feature Dimensionality
                    2: Label Dimensionality
    """
    print "Started Loading"
    f = open(filename, mode='rb')

    header_info = False
    if header:
        header_info = f.readline().split()
        Xsparse, y = load_svmlight_file(
            f=f,
            multilabel=True,

        )
        if concatbias:
            one = csr_matrix(np.ones(shape=(Xsparse.shape[0], 1)))
            X = scipy.sparse.hstack([one, Xsparse], format="csr")
        else:
            X = Xsparse

        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        # Convert y to sparse array, Note : MultiLabelBinarizer() could be used
        ult = (np.zeros((DATASET_SIZE, LABEL_NUMBER)))
        for i in xrange(0, DATASET_SIZE):
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
    print "Finished Loading"
    return X, y, header_info


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


def grad_check(X, W, y):
    """
    Checks the validation of gradient versus a numerical approximation
    :param X: Training examples
    :param W: Weight vector
    :param y: True Categories of the training examples X
    """
    if scipy.sparse.issparse(X):
        true_gradient = gradient_sp(X=X, W=W, y=y)
    else:
        true_gradient = gradient(X=X, W=W, y=y)

    epsilon = 1e-6
    num_grad = np.zeros(W.shape[0])
    iterr = tqdm.trange(0, W.shape[0])
    # iterr = np.arange(0, W.shape[0])
    for k in (iterr):
        W_tmpP = np.zeros((W.shape))
        W_tmpP[:] = W

        W_tmpM = np.zeros(((W.shape)))
        W_tmpM[:] = W

        W_tmpP[k] = W_tmpP[k] + (epsilon)
        if scipy.sparse.issparse(X):
            Ewplus = log_likelihood_sp(X, W_tmpP, y)
        else:
            Ewplus = log_likelihood(X, W_tmpP, y)

        W_tmpM[k] = W_tmpM[k] - (epsilon)
        if scipy.sparse.issparse(X):
            Ewminus = log_likelihood_sp(X, W_tmpM, y)
        else:
            Ewminus = log_likelihood(X, W_tmpM, y)


        num_grad[k] = np.divide((np.subtract(Ewplus, Ewminus)), np.multiply(2, epsilon))
    true_gradient.reshape((W.shape[0], 1))
    num_grad.reshape((W.shape[0], 1))

    absmax = np.linalg.norm(true_gradient - num_grad) / np.linalg.norm(true_gradient + num_grad)
    if absmax > 0.5 * 1e-2:
        return 'WARNING: Failed Gradient Check ' + str(absmax)
    return 'INFO: Successful Gradient Check ' + str(absmax)


def plot_linreg_results(X, W, y, preds, lossHistory):
    # print("Score " + str(accuracy_score(y_true=y, y_pred=preds)))

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


def size(x, str1=' '):
    """
    Prints shape of x along with a message.
    :param x:
    :param str1: A message to print with the shape.
    """
    print(str(x.shape) + ' ' + str1)
    return 0


def tic():
    """
    MATLAB's tic.
    """
    return time.time()


def toc(start_time, str1=" "):
    """
    MATLAB's toc.
    :param start_time: Time supplied by tic
    :param str1: Extra string to be included in the beginning of print statement
    """
    print(str1 + " " + "--- %s seconds ---" % (time.time() - start_time))


def typu(x):
    """
    Print type of object.
    :param x:
    :return:
    """
    print(type(x))
