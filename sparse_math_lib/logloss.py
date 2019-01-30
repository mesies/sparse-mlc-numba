import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix

import sp_operations as sp_op
from sparse_math_lib.mathutil import sigmoid

"""
Log loss implementation.
"""



# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler


# Optimised
def log_likelihood_sp(X, W, y):
    result = log_likelihood_sp_numba_wrapper(X, W, y)
    assert result is not np.nan

    return result


def log_likelihood_sp_old(X, W, y):
    """
    Log loss sparse optimised function.
    @param X: Training examples
    @param W: Weight vector
    @param y: True Categories of the training examples X
    @return: logarithmic loss
    """
    # -1 ^ y Ew = np.sum(t*np.log(s)+(1-t)*np.log(1-s))

    signus = np.ones(y.shape)
    if y.nnz != 0:
        result_row, result_col = sp_op.nonzero(y)
        signus[result_row] = -1

    # (XW) * (-1 ^ y)
    xw = X.dot(W)
    xw_hat = sp_op.mult_col(xw, signus)

    logg = np.logaddexp(0, xw_hat)

    result = np.sum(logg[:, None], axis=0)
    assert result.shape[0] == 1

    return result


def log_likelihood_sp_numba_wrapper(X, W, y):
    xw = X.dot(W)

    signus = np.ones(y.shape, dtype=float).flatten()

    xw_hat = np.zeros((xw.shape), dtype=float)

    return log_likelihood_sp_numba(xw, xw_hat, y.indices, y.indptr, np.max(y.shape), y.data.shape[0], signus)


@numba.jit(['float64(float64[:,:], float64[:,:], int32[:], int32[:], int64, int64, float64[:])',
            'float64(float64[:,:], float64[:,:], int32[:], int32[:], int64, int64, float64[:])'
            ],
           nopython=True,
           nogil=True,
           cache=True,
           fastmath=True
           )
def log_likelihood_sp_numba(xw_2d, xw_hat, y_ind_1d, y_indptr_1d, y_shape_2d, y_data_size_1d, signus_1d):
    """
    Log loss sparse optimised function.
    @param X: Training examples
    @param W: Weight vector
    @param y: True Categories of the training examples X
    @return: logarithmic loss
    """

    result_indrow_1d = np.zeros((y_data_size_1d), dtype=np.int64)
    result_indcol_1d = np.zeros((y_data_size_1d), dtype=np.int64)

    sp_op.nonzero_numba(result_indrow_1d, result_indcol_1d, y_ind_1d, y_indptr_1d, y_shape_2d,
                        np.ones((1), dtype=np.int64)[0])

    signus_1d[result_indrow_1d] = -1

    ########################

    sp_op.mult_col_matrix_numba(signus_1d, xw_2d, xw_hat, xw_2d.shape[0], xw_2d.shape[1])

    # xw_hat = sp_op.mult_col(xw, signus)
    ##############################

    logg = np.logaddexp(0, xw_hat)

    result = np.sum(logg)

    return result

def log_likelihood_numba(X, W, y):
    # Ew = np.sum(t*np.log(s)+(1-t)*np.log(1-s))
    # sdotp is dense
    # y is sparse -> we have indexes

    sdotp = sigmoid(X.dot(W))

    ind_i, ind_y = sp_op.nonzero(y)
    return log_likelihood_numba_n(sdotp, ind_i, ind_y)


@numba.jit(['boolean(int64, int64[:])',
            'boolean(int32, int32[:])'],
           nopython=True,
           nogil=True,
           cache=True,
           fastmath=True)
def contains_numba(value, array):
    for i in range(0, array.shape[0]):
        if array[i] == value: return True

    return False


@numba.jit(['float64(float64[:,:], int64[:], int64[:])',
            'float64(float64[:,:], int32[:], int32[:])'],
           nopython=True,
           nogil=True,
           cache=True,
           fastmath=True
           )
def log_likelihood_numba_n(sdotp, y_indexes_i, y_indexes_y):
    # Ew = np.sum(t*np.log(s)+(1-t)*np.log(1-s))

    result = 0.
    for i in range(0, sdotp.shape[0]):
        for j in range(0, sdotp.shape[1]):
            if contains_numba(i, y_indexes_i) and contains_numba(j, y_indexes_y):
                result += np.log(sdotp[i, j])
            else:
                result += np.log(1 - sdotp[i, j])
            result += np.log(sdotp[i, j])
    return result



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
    AND
    1 - sigmoid(x) = sigmoid(-x)
    =>
    Yn = log(sigmoid(X*W)) if t = 1
    Yn = log(sigmoid((-1)*X*W) if t = 0
    =>
    Yn = log(sigmoid((-1)^(t-1) * (X*W))
    AND
    log(sigmoid(x)) = log(1 / 1 + exp(-x)) = -log(1 + exp(-x))
    =>
    L = -np.sum(-np.log(1 + np.exp(-((-1)**(1-t)) * np.dot(X, W))))

    This functions returns the log likelihood that will be maximized.
    @param X: Training examples
    @param W: Weight vector
    @param y: True Categories of the training examples X
    @return: minus log likelihood
    """
    if scipy.sparse.issparse(X) or scipy.sparse.issparse(W) or scipy.sparse.issparse(y):
        """
        L = - sum(Yn)
        Yn = log(sigmoid(X*W)) if t = 1
        Yn = log(1 - sigmoid(X*W) if t = 0        
        """
        print("WARN : Use log_likelihood_sp for sparse matrices")
        return log_likelihood_sp(X, W, y)

    else:
        sign = (-1) ** y
        xw_hat = sign * np.dot(X, W)
        L = -np.sum(-np.logaddexp(0, xw_hat))

    return L
