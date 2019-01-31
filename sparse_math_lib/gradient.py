import numba
import numpy as np
import scipy.sparse

import sp_operations as sp_op
from mathutil import sigmoid

"""
This file implements the gradient.
"""


# profile = lambda f: f
def gradient_sp(X, W, y):
    return gradient_sp_numba_wrapper(X, W, y)


# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler

# @profile
def gradient_sp_old(X, W, y):
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
        # Because y[ind] = 1, iff ind = y.nonzero()
        resultrow, resultcol = sp_op.nonzero(y)
        sdotp[resultrow] -= 1

    # (sigm(XW) - y) * X,T
    # data, row, col = mult_row_raw(X, sdotp)

    result = sp_op.mult_col_raw_col_row_sum_raw(X, sdotp)
    assert result.shape[1] == W.shape[1]

    return result


def gradient_sp_numba_wrapper(X, W, y):
    xw_1d = X.dot(W)  # dense
    result_1d = np.zeros(np.max(W.shape))

    y_data_size_1d = y.data.shape[0]

    result_indrow_1d = np.zeros((y_data_size_1d), dtype=np.int64)
    result_indcol_1d = np.zeros((y_data_size_1d), dtype=np.int64)

    return gradient_numba(
        result_1d,
        result_indrow_1d,
        result_indcol_1d,
        X.shape[0],
        X.shape[1],
        X.data,
        X.indices,
        X.indptr,
        xw_1d,
        y.indices,
        y.indptr,
        np.array(y.shape)
    ).reshape((result_1d.shape[0], 1))


@numba.jit(['float64[:]('
            'float64[:], '
            'int64[:], '
            'int64[:], '
            'int64,'
            'int64,'
            'float64[:], '
            'int32[:], '
            'int32[:], '
            'float64[:,:], '
            'int32[:], '
            'int32[:],'
            'int32[:]'
            ')'
            ],
           nopython=True,
           nogil=True,
           cache=True,
           fastmath=True
           )
def gradient_numba(
        result_1d,
        result_indrow_1d,
        result_indcol_1d,
        x_shape0,
        x_shape1,
        x_data,
        x_indices,
        x_indptr,
        xw_1d,
        y_ind_1d,
        y_indptr_1d,
        y_shape_2d
):
    sp_op.nonzero_numba(result_indrow_1d, result_indcol_1d, y_ind_1d, y_indptr_1d, np.max(y_shape_2d),
                        np.ones((1), dtype=np.int64)[0])

    # A, B = sp_op.nonzero(y)
    # assert np.array_equal(A, result_indrow_1d)
    # assert np.array_equal(B, result_indcol_1d)

    row_vector = sigmoid(xw_1d)

    row_vector[result_indrow_1d] -= 1

    # result = sp_op.mult_col_raw_col_row_sum_raw(X, row_vector)

    result_data = x_data.copy()
    result_row = np.zeros(x_data.shape[0], dtype=np.int64)
    result_col = np.zeros(x_data.shape[0], dtype=np.int64)

    sp_op.mult_col_raw_col_row_sum_raw_numba(
        result_1d,
        row_vector.ravel(),
        x_indptr,
        x_indices,
        result_data,
        result_row,
        result_col,
        x_shape0,
        x_shape1)

    # assert np.array_equal(result, result_1d.reshape((result_1d.shape[0], 1)))

    return result_1d


def gradient(X, W, y):
    """
           Gradient of log_likelihood
           @param X: Training examples
           @param W: Weight vector
           @param y: True Categories of the training examples X
           @return: Gradient
           """

    if scipy.sparse.issparse(X) or scipy.sparse.issparse(W) or scipy.sparse.issparse(y):
        print("WARN : Use gradient_sp for sparse matrices")
        return gradient_sp(X, W, y)

    sig = (sigmoid(np.dot(X, W))).T - y
    assert sig.shape == y.shape

    inss = sig * X.T
    assert inss.shape == X.T.shape

    result = np.sum(inss, axis=1)
    assert result.shape[0] == inss.shape[0]

    return result
