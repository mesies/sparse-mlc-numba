# import autograd
# import autograd.numpy as np
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
import tqdm
from mathutil import gradient, log_likelihood, gradient_sp, log_likelihood_sp

# Uncomment when debugging with line_profiler
profile = lambda f: f

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
    print("Started Loading Dataset")
    f = open(filename, mode='rb')

    header_info = False
    if header:
        header_info = f.readline().split()
        Xsparse, y = load_svmlight_file(
            f=f,
            multilabel=True,

        )
        if concatbias:
            one = sp.csr_matrix(np.ones(shape=(Xsparse.shape[0], 1)))
            X = scipy.sparse.hstack([one, Xsparse], format="csr")
        else:
            X = Xsparse

        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        ult = (sp.lil_matrix((DATASET_SIZE, LABEL_NUMBER)))
        for i in tqdm.trange(0, DATASET_SIZE):
            ind_of_labels = np.asarray(y[i], dtype=int)
            ult[i, ind_of_labels] = 1
        y = ult.asformat('csr')

    else:
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,
        )
    f.close()
    print("Finished Loading")
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
    for k in iterr:
        W_tmpP = np.zeros(W.shape)
        W_tmpP[:] = W

        W_tmpM = np.zeros((W.shape))
        W_tmpM[:] = W

        W_tmpP[k] = W_tmpP[k] + epsilon
        if scipy.sparse.issparse(X):
            Ewplus = log_likelihood_sp(X, W_tmpP, y)
        else:
            Ewplus = log_likelihood(X, W_tmpP, y)

        W_tmpM[k] = W_tmpM[k] - epsilon
        if scipy.sparse.issparse(X):
            Ewminus = log_likelihood_sp(X, W_tmpM, y)
        else:
            Ewminus = log_likelihood(X, W_tmpM, y)

        num_grad[k] = np.divide((np.subtract(Ewplus, Ewminus)), np.multiply(2, epsilon))
    true_gradient.reshape((W.shape[0], 1))
    num_grad.reshape((W.shape[0], 1))

    abs_max = np.linalg.norm(true_gradient - num_grad) / np.linalg.norm(true_gradient + num_grad)
    if abs_max > 0.5 * 1e-2:
        return 'WARNING: Failed Gradient Check ' + str(abs_max)
    return 'INFO: Successful Gradient Check ' + str(abs_max)


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


def save_sparse_csr(filename, array):
    """
    Saves a sparse array in npz format in current directory.
    :param filename:
    :param array:
    :return:
    """
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    """
    Loads a sparse array from an npz file in current directory.
    :param filename:
    :return:
    """
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


@profile
def concatenate_csr_matrices_by_columns(matrix1, matrix2):
    """
    Concatenates two csr sparse matrices in a more efficient way than hstack
    :param matrix1:
    :param matrix2:
    :return:
    """
    csr = isinstance(matrix1, scipy.sparse.csr_matrix)
    if csr:
        matrix1 = matrix1.T.asformat('csr')
        matrix2 = matrix2.T.asformat('csr')
    else:
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    if csr:
        return sp.csr_matrix(
            (new_data,
             new_indices,
             new_ind_ptr)
        ).T.asformat('csr')
    else:
        return sp.csr_matrix(
            (new_data,
             new_indices,
             new_ind_ptr)
        ).T.asformat('csc')
