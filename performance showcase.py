import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import tqdm

import helpers
import sparse_math_lib.sp_operations


# Nonzero
def runtime_x_density():
    m = 1000
    n = 10000
    iter = 100
    X = np.arange(0.001, 0.001 * iter, 0.001)
    Y = []
    Y2 = []

    for i in tqdm.tqdm(np.arange(0.001, 0.001 * iter, 0.001)):
        A = scipy.sparse.rand(m, n, density=i, format='csr', dtype=float)
        t = helpers.tic()
        sparse_math_lib.sp_operations.nonzero(A)
        Y.append(helpers.toc(t))

        t = helpers.tic()
        A.nonzero()
        Y2.append(helpers.toc(t))

    plt.plot(X, Y, 'r', label='Custom Numba nonzero function')
    plt.plot(X, Y2, 'c', label='scipy.sparse nonzero function')
    plt.xlabel('Density of matrix')
    plt.ylabel('time')
    plt.legend(loc="lower right")

    plt.title("Function runtime as density increases")
    plt.show()


def runtime_x_columns():
    m = 1000
    n = 10000
    iter = 10

    n_batch = 5000
    X = np.arange(n_batch, n_batch * iter, n_batch)
    Y = []
    Y2 = []

    for i in tqdm.tqdm(X):
        A = scipy.sparse.rand(n, i, density=0.001, format='csr', dtype=float)
        t = helpers.tic()
        sparse_math_lib.sp_operations.nonzero(A)
        Y.append(helpers.toc(t))

        t = helpers.tic()
        A.nonzero()
        Y2.append(helpers.toc(t))

    plt.plot(X, Y, 'r', label='Custom Numba nonzero function')
    plt.plot(X, Y2, 'c', label='scipy.sparse nonzero function')
    plt.xlabel('columns of matrix')
    plt.ylabel('time')
    plt.legend(loc="lower right")

    plt.title("Function runtime as columns of matrix are increasing")
    plt.show()


def runtime_x_rows():
    m = 1000
    n = 10000
    iter = 10

    n_batch = 5000
    X = np.arange(n_batch, n_batch * iter, n_batch)

    Y = []
    Y2 = []

    for i in tqdm.tqdm(X):
        A = scipy.sparse.rand(i, n, density=0.001, format='csr', dtype=float)
        t = helpers.tic()
        sparse_math_lib.sp_operations.nonzero(A)
        Y.append(helpers.toc(t))

        t = helpers.tic()
        A.nonzero()
        Y2.append(helpers.toc(t))

    plt.plot(X, Y, 'r', label='Custom Numba nonzero function')
    plt.plot(X, Y2, 'c', label='scipy.sparse nonzero function')
    plt.xlabel('rows of matrix')
    plt.ylabel('time')
    plt.legend(loc="lower right")

    plt.title("Function runtime as rows of matrix are increasing")
    plt.show()


# runtime_x_density()
# runtime_x_columns()
# runtime_x_rows()
sparse_math_lib.sp_operations.nonzero_numba.inspect_types()
