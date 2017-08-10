import numba
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix


def sigmoid(x):
    """
    Sigmoid function.
    """

    # result = 1. / (1. + np.exp(-x))
    # result = 0.5 * (np.tanh(0.5 * x) + 1)
    result = scipy.special.expit(x)
    return result

def sigmoid_sp(x):
    result = np.zeros(x.shape)
    r = scipy.special.expit(x.data)

    return result

def gradient_sp(X, W, y):
    """
       Gradient of log_likelihood
       :param X: Training examples
       :param W: Weight vector
       :param y: True Categories of the training examples X
       :return: Gradient
       """
    dott = X.dot(W)
    dotp = sigmoid(dott)

    y = y.toarray()
    ylike = y.reshape((y.shape[0]))

    sdotp = dotp.T - ylike
    sdotp = sdotp.reshape((sdotp.shape[0], 1))

    ihh = np.repeat(sdotp, X.shape[1], axis=1)
    inss = X.multiply(ihh)

    result = np.sum(inss, axis=0).A1

    return result


def log_likelihood_sp(X, W, y):

    signus = np.ones(y.shape)
    signus[y.nonzero()] = -1

    dotr = X.dot(csr_matrix(W).T)

    xw_hat = (dotr).multiply(signus)
    # Wrong
    logg = -np.logaddexp(0, xw_hat.toarray())
    L = -np.sum(logg)
    return L

@numba.jit
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

@numba.jit
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
        sign = (-1) ** (1 - y)
        xw_hat = sign * np.dot(X, W)
        L = -np.sum(-np.logaddexp(0, (-xw_hat)))

    return L