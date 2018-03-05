import numpy as np

from sparse_math_lib.mathutil import sigmoid
from sparse_math_lib.sp_operations import nonzero, coo_row_sum

"""
This file implements the gradient.
"""
profile = lambda f: f


@profile
def gradient_sp(X, W, y):
    """
       Gradient of log_likelihood optimised for sparse matrices.
       :param X: Training examples
       :param W: Weight vector
       :param y: True Categories of the training examples X
       :return: Gradient
       """
    # sigm(XW) - y
    sdotp = sigmoid(X.dot(W))

    # ############################################Doable
    if y.nnz != 0:
        # Originally it is required to compute
        #             s = -1 ^ (1 - y_n)
        #
        # which for : y_n = 1 -> s = 1
        # and for   : y_n = 0 -> s = -1
        # Because y[ind] = 1, if ind = y.nonzero()
        resultrow, resultcol = nonzero(y)
        sdotp[resultrow] -= 1
    # #############################################
    # # (sigm(XW) - y) * X,T
    # d = X.toarray()
    in_sum = X.multiply(sdotp)
    # in_sum = sp.csc_matrix(in_sum)
    #in_sum = np.multiply(d, sdotp)
    ############################################Doable
    # result = np.zeros(X.shape[1], dtype=float)
    # sum_rows_of_matrix_numba(in_sum, result, in_sum.shape[0], in_sum.shape[1])
    # result = np.sum(in_sum, axis=0).A
    # result = (in_sum).sum(axis=0).A # Best
    result = coo_row_sum(in_sum)
    # assert result.shape == W_1dim.shape

    return result.T
    ###########################################


def gradient(X, W, y):
    """
           Gradient of log_likelihood
           :param X: Training examples
           :param W: Weight vector
           :param y: True Categories of the training examples X
           :return: Gradient
           """
    sig = (sigmoid(np.dot(X, W))).T - y
    assert sig.shape == y.shape

    inss = sig * X.T
    assert inss.shape == X.T.shape

    result = np.sum(inss, axis=1)
    assert result.shape[0] == inss.shape[0]

    return result
