import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix


# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler
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
    dotp = sigmoid(X.dot(W))

    sdotp = dotp.T
    if y.nnz != 0:
        resultrow, resultcol = nonzero(y)
        sdotp[resultrow] -= 1  # Because y[ind] = 1, if ind = y.nonzero()


    # (sigm(XW) - y) * X,T

    # in_sum = np.zeros((X.shape))
    # indrow, indcol = nonzero(X)
    # in_sum[indrow, indcol] = np.take(sdotp, indrow, axis=0)


    # idea : csr -> data tou i row data[indptr[:i],indprt[i+1]

    # (sigm(XW) - y) * X,T
    in_sum = X.multiply(sdotp[:, np.newaxis]).A
    # in_sum = sparse_mult(X, sdotp)

    # sum()
    result = np.zeros(X.shape[1], dtype=float)
    # sum_rows(in_sum, result, in_sum.shape[0], in_sum.shape[1])
    result = np.sum(in_sum, axis=0)

    return result


@numba.jit('void(float64[:,:], float64[:], int64, int64)',
           nopython=True,
           cache=True,
           nogil=True)
def sum_rows(x, result, dim0, dim1):
    """
    Numba function which returns sum of eaxh row e.g. [1 2 3] -> 6 [1 1],[2 2],[3 3] -> [2], [4], [6]
    :param x:
    :param result:
    :param dim0:
    :param dim1:
    :return:
    """
    for i in range(0, dim0):
        for j in range(0, dim1):
            result[j] += x[i, j]


# Optimised
def log_likelihood_sp(X, W, y):
    """
    Log loss sparse optimised function.
    :param X:
    :param W:
    :param y:
    :return:
    """

    # -1 ^ y
    signus = np.ones(y.shape)
    if y.nnz != 0:
        resultrow, resultcol = nonzero(y)
        signus[resultrow] = -1

    # (XW) * (-1 ^ y)
    dotr = X.dot(W)
    xw_hat = np.zeros(signus.shape)
    multt(dotr, signus, xw_hat, signus.shape[0], signus.shape[1])

    logg = np.logaddexp(0, xw_hat)

    # Minus applied on summ function
    result = 0.
    summ(result, -logg[:, 0], logg.shape[0])
    # result = np.sum(logg[:, None], axis=0)
    return result


def sparse_mult(x, y_rows):
    """
    Element-wise multiplication of a sparse array by a row of values, meaning each row of sparse array x is multiplied
    element-wise by y_rows.
    Wrapper for special_mult
    :param x:
    :param y_rows:
    :return:
    """
    result = x.toarray()
    special_mult(result,
                 x.data,
                 x.indices,
                 x.indptr,
                 x.shape[0],
                 y_rows,
                 np.zeros(x.data.shape[0], dtype=int),
                 np.zeros(x.data.shape[0], dtype=int)
                 )
    return result


@numba.jit('void(float64[:,:], '
           'float64[:],'
           'int32[:], '
           'int32[:], '
           'int32,'
           'float64[:], '
           'int32[:], '
           'int32[:])',
           nopython=True,
           nogil=True)
def special_mult(result,
                 x_data,
                 x_indices,
                 x_indptr,
                 x_shape0,
                 y,
                 z_indrow,
                 z_indcol
                 ):
    """
    See sparse_mult
    :param result:
    :param x_data:
    :param x_shape0:
    :param z_indrow:
    :param z_indcol:
    :param x_data:
    :param x_indices:
    :param x_indptr:
    :param y:
    :return:
    """
    indrow = z_indrow
    indcol = z_indcol

    # Same code as nonzero_numb because of a known numba bug
    h = 0
    for i in range(0, x_shape0):
        ind = x_indices[x_indptr[i]:x_indptr[i + 1]]
        for j in ind:
            indrow[h] = i
            indcol[h] = j
            h += 1

    for k in range(0, len(indrow)):
        i = indrow[k]
        j = indcol[k]
        result[i, j] = result[i, j] * y[j]


def nonzero(x):
    """
    Returns indices of non zero elements of a scipy csr_matrix, wrapper for nonzero_numb
    :param x: The matrix in question
    :return: row indices and column indices
    """
    indrow = np.zeros((x.data.shape[0]), dtype=int)
    indcol = np.zeros((x.data.shape[0]), dtype=int)
    if isinstance(x, csr_matrix):
        nonzero_numb(indrow, indcol, x.data, x.indices, x.indptr, x.shape[0], 1)
    elif isinstance(x, csc_matrix):
        nonzero_numb(indrow, indcol, x.data, x.indices, x.indptr, x.shape[1], 0)
    return indrow, indcol


@numba.jit('void(int32[:],int32[:], float64[:], int32[:], int32[:], int32, int32)',
           nopython=True,
           nogil=True)
def nonzero_numb(resultrow, resultcol, data, indices, indptr, columns, isitcsr):
    """
    See nonzero
    :param resultrow:
    :param resultcol:
    :param data:
    :param indices:
    :param indptr:
    :param columns:
    :param isitcsr:
    :return:
    """
    # column indices for column i is in indices[indptr[i]:indptr[i+1]]
    if isitcsr == 1:
        h = 0
        for i in range(0, columns):
            ind = indices[indptr[i]:indptr[i + 1]]
            for j in ind:
                resultrow[h] = i
                resultcol[h] = j
                h += 1
    else:
        h = 0
        for i in range(0, columns):
            ind = indices[indptr[i]:indptr[i + 1]]
            for j in ind:
                resultrow[h] = j
                resultcol[h] = i
                h += 1


@numba.jit('void(float64, float64[:], int64)',
           nopython=True,
           cache=True,
           nogil=True)
def summ(result, x, sh):
    """
    An optimised summation using Numba's JIT compiler
    :param result:
    :param x:
    :param sh:
    :return:
    """
    s = x[0]
    for i in range(1, sh):
        s = s + x[i]
    result = -s


@numba.jit('void(float64[:], float64[:,:], float64[:,:], int64, int64)',
           nopython=True,
           cache=True,
           nogil=True)
def multt(row_matrix, matrix, result, dim0, dim1):
    """
    Optimised matrix element-wise multiplication when one matrix
    :param row_matrix:
    :param matrix:
    :param result:
    :param dim0:
    :param dim1:
    :return:
    """
    for i in range(0, dim0):
        for j in range(0, dim1):
            result[i, j] = row_matrix[i] * matrix[i, j]


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
    if scipy.sparse.issparse(X):
        """
        L = - sum(Yn)
        Yn = log(sigmoid(X*W)) if t = 1
        Yn = log(1 - sigmoid(X*W) if t = 0        
        """
        signus = np.ones(y.shape)
        signus[y.nonzero()] = -1

        dotr = X.dot(csr_matrix(W).T)

        xw_hat = dotr.multiply(signus)

        logg = -np.logaddexp(0, xw_hat.toarray())
        L = -np.sum(logg)

    else:
        sign = (-1) ** y
        xw_hat = sign * np.dot(X, W)
        L = -np.sum(-np.logaddexp(0, xw_hat))

    return L

@numba.vectorize
def sigmoid(x):
    """
    Sigmoid function.
    """

    # result = 1. / (1. + np.exp(-x))
    result = 0.5 * (np.tanh(0.5 * x) + 1)
    # result = scipy.special.expit(x)
    return result
