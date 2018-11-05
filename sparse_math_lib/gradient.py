import numpy as np

from sparse_math_lib.mathutil import sigmoid
from sparse_math_lib.sp_operations import nonzero, col_row_sum_raw_mult_col_raw

"""
This file implements the gradient.
"""
profile = lambda f: f


# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler

@profile
def gradient_sp(X, W, y):
    """
       Gradient of log_likelihood optimised for sparse matrices.
       @param X: Training examples
       @param W: Weight vector
       @param y: True Categories of the training examples X
       @return: Gradient
       """
    # sigm(XW) - y
    sdotp = sigmoid(X.dot(W))

    if y.nnz != 0:
        # Originally it is required to compute
        #             s = -1 ^ (1 - y_n)
        #
        # which for : y_n = 1 -> s = 1
        # and for   : y_n = 0 -> s = -1
        # Because y[ind] = 1, if ind = y.nonzero()
        resultrow, resultcol = nonzero(y)
        sdotp[resultrow] -= 1

    # (sigm(XW) - y) * X,T
    # data, row, col = mult_row_raw(X, sdotp)
    # result = col_row_sum_raw(data, row, col, X.shape[0], X.shape[1])
    result = col_row_sum_raw_mult_col_raw(X, sdotp)
    assert result.shape[0] == W.shape[1]

    return result.T


def gradient(X, W, y):
    """
           Gradient of log_likelihood
           @param X: Training examples
           @param W: Weight vector
           @param y: True Categories of the training examples X
           @return: Gradient
           """
    sig = (sigmoid(np.dot(X, W))).T - y
    assert sig.shape == y.shape

    inss = sig * X.T
    assert inss.shape == X.T.shape

    result = np.sum(inss, axis=1)
    assert result.shape[0] == inss.shape[0]

    return result
