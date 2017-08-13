import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import numexpr as ne
#profile = lambda f: f


def sigmoid(x):
    """
    Sigmoid function.
    """

    # result = 1. / (1. + np.exp(-x))
    # result = 0.5 * (np.tanh(0.5 * x) + 1)
    result = scipy.special.expit(x)
    return result

@profile
def gradient_sp(X, W, y, result=''):
    """
       Gradient of log_likelihood
       :param X: Training examples
       :param W: Weight vector
       :param y: True Categories of the training examples X
       :return: Gradient
       """
    # Optimised : X * W is dense
    dott = X.dot(W)
    #dotp = (ne.evaluate('1 / (1 + exp(-dott))'))
    dotp = sigmoid(dott)


    if y.nnz != 0:
        ind = y.nonzero()[0]
        dotp.T[ind] -= 1   #Because y[ind] = 1, if ind = y.nonzero()
    sdotp = dotp.T

    inss = np.zeros((X.shape))
    ind = X.nonzero()
    inss[ind[0], ind[1]] = np.take(sdotp, ind[0], axis=0)

    result = np.sum(inss, axis=0)
    #result = ne.evaluate('sum(inss, axis=0)')

    return result


@profile
def log_likelihood_sp(X, W, y, result=''):

    signus = np.ones(y.shape)
    if y.nnz != 0:
        signus[y.nonzero()] = -1
        #signus_numba(signus, y.nonzero()[0], len(y.nonzero()[0]))

    # y -> 1 = -1 y -> 0 -> 1

    dotr = X.dot(W)
    #dotr = dotr.reshape((dotr.shape[0], 1))

    #xw_hat = np.multiply(dotr, signus)
    xw_hat = np.zeros(signus.shape)
    xw_hat = multt(dotr, signus, xw_hat, signus.shape[0], signus.shape[1])
    #xw_hat = ne.evaluate('dotr * signus')

    #logg = ne.evaluate('-log(1 + exp(xw_hat))')
    logg = -np.logaddexp(0, xw_hat)

    #Minus applied on summ function
    #L = summ(logg[:,0], logg.shape[0])
    #L = ne.evaluate('sum(logg)')
    #L = -np.sum(logg)

    result = summ(logg[:,0], logg.shape[0])
    return result


@numba.jit('void(float64[:,:], int32[:], int64)', nopython = True, cache=True)
def signus_numba(y1, indexes, sh):
        y1[indexes] = -1

@numba.jit('float64(float64[:], int64)',nopython=True, cache=True)
def summ(x, sh):
    s = x[0]
    for i in range(1, sh):
        s = s + x[i]
    return -s

@numba.jit('float64[:,:](float64[:], float64[:,:], float64[:,:], int64, int64)',nopython=True, cache=True)
def multt(A, B, C, dim0, dim1):
    for i in range(0, dim0):
        for j in range(0, dim1):
            C[i,j] = A[i] * B[i,j]
    return C

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