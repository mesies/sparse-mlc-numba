import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import tqdm

import helpers
import sparse_math_lib.sp_operations


# Nonzero
def runtime_x_density():
    m = 10000
    n = 100000
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
    m = 10000
    n = 100000
    iter = 10
    X = np.arange(10000, 10000 * 2, (10000 * 2) / 10)
    Y = []
    Y2 = []

    for i in tqdm.tqdm(np.arange(10000, 10000 * 2, (10000 * 2) / 10)):
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
    m = 10000
    n = 100000
    iter = 10
    X = np.arange(10000, 10000 * 2, (10000 * 2) / 10)
    Y = []
    Y2 = []

    for i in tqdm.tqdm(np.arange(10000, 10000 * 2, (10000 * 2) / 10)):
        A = scipy.sparse.rand(n, i, density=0.001, format='csr', dtype=float)
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
