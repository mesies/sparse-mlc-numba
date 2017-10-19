import numpy as np
import scipy
from scipy.sparse import csr_matrix

from sparse_math_lib.sp_operations import nonzero, mult_col_matrix_numba, sum_of_vector_numba

profile = lambda f: f


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
    mult_col_matrix_numba(xw, signus, xw_hat, signus.shape[0], signus.shape[1])

    logg = np.logaddexp(0, xw_hat)

    # Minus applied on summ function
    result = 0.
    result = sum_of_vector_numba(result, -logg[:, 0], logg.shape[0])
    # result = np.sum(logg[:, None], axis=0) - 0.5 * 0.01 * np.linalg.norm(W)
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
