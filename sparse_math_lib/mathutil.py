import numba
import numpy as np


# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler


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
