import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler
profile = lambda f: f


def ult_grad(X, W, y):
    dotp = sigmoid(X.dot(W))
    sdotp = dotp.T

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

    #############################################Doable
    if y.nnz != 0:
        # Originally it is required to compute
        #             s = -1 ^ (1 - y_n)
        # which for y_n = 1 , s = 1
        # and for   y_n = 0 , s = -1
        # Because y[ind] = 1, if ind = y.nonzero()
        resultrow, resultcol = nonzero(y)
        sdotp[resultrow] -= 1
    ##############################################

    # # (sigm(XW) - y) * X,T
    # G = X.tocsc()
    ### SCIPY makes X dense in mult https://github.com/scipy/scipy/blob/v0.16.1/scipy/sparse/compressed.py#L394
    in_sum = X.multiply((sdotp[:, np.newaxis])).A
    # in_sum = mult_row_sparse(X, sdotp)
    # in_sum = np.multiply(sdotp[:, np.newaxis], X.toarray())

    # Request nnz, use for mult
    ############################################Doable
    result = np.zeros(X.shape[1], dtype=float)
    sum_rows(in_sum, result, in_sum.shape[0], in_sum.shape[1])
    # result = np.sum(in_sum, axis=0) # - 0.01 * np.linalg.norm(W)
    return result
    ###########################################


def mult_row_sparse(A, B):
    resultrow, resultcol = nonzero(A)
    result = A.toarray()
    mult_row_sparse_numba(result, B.ravel(), resultrow, resultcol, resultrow.shape[0])
    return result


@numba.jit('void(float64[:,:], float64[:], int32[:], int32[:], int64)',
           nopython=True
           )
def mult_row_sparse_numba(result, B, resultrow, resultcol, lenrow):
    # result init zero

    # print numba.typeof(result)
    # print numba.typeof(B)
    # print numba.typeof(resultrow)
    # print numba.typeof(resultcol)
    # print numba.typeof(lenrow)
    # print numba.typeof(lencol)
    for i in xrange(0, lenrow):
        row = np.take(resultrow, i)
        col = np.take(resultcol, i)
        result[row, col] *= np.take(B, col)

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
        result_row, result_col = nonzero(y)
        signus[result_row] = -1

    # (XW) * (-1 ^ y)
    xw = X.dot(W)
    xw_hat = np.zeros(signus.shape)
    multt(xw, signus, xw_hat, signus.shape[0], signus.shape[1])

    logg = np.logaddexp(0, xw_hat)

    # Minus applied on summ function
    result = 0.
    result = summ(result, -logg[:, 0], logg.shape[0])
    # result = np.sum(logg[:, None], axis=0) - 0.5 * 0.01 * np.linalg.norm(W)
    return result


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
def nonzero_numb(result_row, result_col, data, indices, indptr, columns, iscsr):
    """
    See nonzero
    :param result_row:
    :param result_col:
    :param data:
    :param indices:
    :param indptr:
    :param columns:
    :param iscsr:
    :return:
    """
    # column indices for column i is in indices[indptr[i]:indptr[i+1]]
    if iscsr == 1:
        h = 0
        for i in range(0, columns):
            ind = indices[indptr[i]:indptr[i + 1]]
            for j in ind:
                result_row[h] = i
                result_col[h] = j
                h += 1
    else:
        h = 0
        for i in range(0, columns):
            ind = indices[indptr[i]:indptr[i + 1]]
            for j in ind:
                result_row[h] = j
                result_col[h] = i
                h += 1


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


@numba.jit('float64(float64, float64[:], int64)',
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
    return result


def mult_row(A, B):
    result = np.zeros(A.shape)
    multt2(B.ravel(), A.toarray(), result, A.shape[0], A.shape[1])
    return result

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


@numba.jit('void(float64[:], float64[:,:], float64[:,:], int64, int64)',
           nopython=True,
           cache=True,
           nogil=True)
def multt2(row_matrix, matrix, result, dim0, dim1):
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
            result[i, j] = row_matrix[j] * matrix[i, j]

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


@numba.vectorize(['float64(float64)'], target='cpu')
def sigmoid(x):
    """
    Sigmoid function.
    """

    # if scipy.sparse.issparse(x):
    #     return csr_matrix(sigmoid(x.data), x.indices, x.indprt)
    # else:
    # result = 1. / (1. + np.exp(-x))
    result = 0.5 * (np.tanh(0.5 * x) + 1)
    # result = scipy.special.expit(x)
    return result
