import numpy as np
import scipy
from scipy.sparse import csr_matrix

import sp_operations as sp_op
from helpers.profile_support import profile

"""
Log loss implementation.
"""



# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler


# Optimised
@profile
def log_likelihood_sp(X, W, y):
    """
    Log loss sparse optimised function.
    @param X:
    @param W:
    @param y:
    @return:
    """
    # -1 ^ y
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
    if scipy.sparse.issparse(X):
        """
        L = - sum(Yn)
        Yn = log(sigmoid(X*W)) if t = 1
        Yn = log(1 - sigmoid(X*W) if t = 0        
        """
        signus = np.ones(y.shape)
        signus[sp_op.nonzero(y)] = -1

        dotr = X.dot(csr_matrix(W).T)

        xw_hat = dotr.multiply(signus)

        logg = -np.logaddexp(0, xw_hat.toarray())
        L = -np.sum(logg)

    else:
        sign = (-1) ** y
        xw_hat = sign * np.dot(X, W)
        L = -np.sum(-np.logaddexp(0, xw_hat))

    return L
