import numba
import numpy as np


@numba.vectorize(['float64(float64)'], target='cpu')
def sigmoid(x):
    """
    Sigmoid function.
    """

    # if scipy.sparse.issparse(x):
    #     return csr_matrix(sigmoid(x.data), x.indices, x.indprt)
    # else:
    # np.clip(x, -500, 500)
    # result = 1. / (1. + np.exp(-x))
    result = 0.5 * (np.tanh(0.5 * x) + 1)
    # result = scipy.special.expit(x)
    return result
