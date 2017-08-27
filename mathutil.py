import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import scipy.sparse as sp
from numba import hsa
#'@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
#as few lines as possible each time line_profiler is run a lambda is used
profile = lambda f: f


@profile
def gradient_sp(X, W, y, result=''):
    """
       Gradient of log_likelihood
       :param X: Training examples
       :param W: Weight vector
       :param y: True Categories of the training examples X
       :return: Gradient
       """


    #sigm(XW) - y
    dotp = X.dot(W)
    dotp = sigmoid(dotp)

    if y.nnz != 0:
        resultrow, resultcol = nonzero(y)
        dotp.T[resultrow] -= 1   #Because y[ind] = 1, if ind = y.nonzero()
    sdotp = dotp.T

    #(sigm(XW) - y) * X,T
    #in_sum = np.zeros((X.shape))
    #indrow, indcol = nonzero(X)
    #in_sum[indrow, indcol] = np.take(sdotp, indrow, axis=0)


    # idea : csr -> data tou i row data[indptr[:i],indprt[i+1]

    #(sigm(XW) - y) * X,T
    in_sum = X.multiply(sdotp[:, np.newaxis]).A

    #result = np.sum(in_sum, axis=0 )
    result = np.zeros(X.shape[1], dtype=float)
    sum_rows(in_sum, result, in_sum.shape[0], in_sum.shape[1])
    #result = np.sum(in_sum, axis=0)

    return result


@numba.jit('void(float64[:,:], float64[:], int64, int64)',
           nopython=True,
           cache=True)
def sum_rows(x, result, dim0, dim1):
    for i in range(0, dim0):
        for j in range(0, dim1):
            result[j] += x[i, j]

#Optimised
def log_likelihood_sp(X, W, y, result=''):

    # -1 ^ y
    signus = np.ones(y.shape)
    if y.nnz != 0:
        resultrow, resultcol = nonzero(y)
        signus[resultrow] = -1

    # (XW) * (-1 ^ y)
    dotr = X.dot(W)
    xw_hat = np.zeros(signus.shape)
    xw_hat = multt(dotr, signus, xw_hat, signus.shape[0], signus.shape[1])

    logg = np.logaddexp(0, xw_hat)

    #Minus applied on summ function
    result = summ(-logg[:, 0], logg.shape[0])
    #result = np.sum(logg[:, None], axis=0)
    return result


def nonzero(x):
    """
    Returns indices of non zero elements of a scipy csr_matrix
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
           cache=True)
def nonzero_numb(resultrow, resultcol, data, indices, indptr, columns, isitcsr):
    # column indices for column i is in indices[indptr[i]:indptr[i+1]]
    if isitcsr == 1:
        h = 0
        for i in range(0, columns):
            ind = indices[indptr[i]:indptr[i+1]]
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


@numba.jit('float64(float64[:], int64)',
           nopython=True,
           cache=True)
def summ(x, sh):
    """
    An optimised summation using Numba's JIT compiler
    :param x:
    :param sh:
    :return:
    """
    s = x[0]
    for i in range(1, sh):
        s = s + x[i]
    return -s


@numba.jit('float64[:,:](float64[:], float64[:,:], float64[:,:], int64, int64)',
           nopython=True,
           cache=True)
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
    return result


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

    inss = (sig) * X.T
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

        xw_hat = (dotr).multiply(signus)

        logg = -np.logaddexp(0, xw_hat.toarray())
        L = -np.sum(logg)

    else:
        sign = (-1) ** (y)
        xw_hat = sign * np.dot(X, W)
        L = -np.sum(-np.logaddexp(0, (xw_hat)))

    return L

@numba.vectorize
def sigmoid(x):
    """
    Sigmoid function.
    """

    # result = 1. / (1. + np.exp(-x))
    result = 0.5 * (np.tanh(0.5 * x) + 1)
    #result = scipy.special.expit(x)
    return result
