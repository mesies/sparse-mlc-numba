import numba
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


@numba.jit('void(float64[:,:], float64[:], int32[:], int32[:], int64)',
           nopython=True
           )
def mult_row_sparse_numba(result, matrix, result_row, result_col, lenrow):
    # result init zero

    # print numba.typeof(result)
    # print numba.typeof(B)
    # print numba.typeof(resultrow)
    # print numba.typeof(resultcol)
    # print numba.typeof(lenrow)
    # print numba.typeof(lencol)
    for i in xrange(0, lenrow):
        row = np.take(result_row, i)
        col = np.take(result_col, i)
        result[row, col] *= np.take(matrix, col)


def nonzero(x):
    """
    Returns indices of non zero elements of a scipy csr_matrix, wrapper for nonzero_numb.
    :param x: The sparse matrix in question only csr and csc matrices are supported
    :return: row indices and column indices
    """

    indrow = np.zeros((x.data.shape[0]), dtype=int)
    indcol = np.zeros((x.data.shape[0]), dtype=int)
    if isinstance(x, csr_matrix):
        nonzero_numba(indrow, indcol, x.indices, x.indptr, x.shape[0], 1)
    elif isinstance(x, csc_matrix):
        nonzero_numba(indrow, indcol, x.indices, x.indptr, x.shape[1], 0)
    else:
        raise NotImplementedError(
            "nonzero is implemented only for csr and csc matrices use scipy.sparse.(any sparse format).nonzero()")
    return indrow, indcol


@numba.jit(['void(int64[:], int64[:], int32[:], int32[:], int64, int64)',
            'void(int32[:], int32[:], int32[:], int32[:], int64, int64)'],
           nopython=True,
           nogil=True,
           cache=True)
def nonzero_numba(result_row, result_col, indices, indptr, columns, iscsr):
    """
    See nonzero
    :param result_row:
    :param result_col:
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
def sum_rows_of_matrix_numba(x, result, dim0, dim1):
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


# noinspection PyUnusedLocal
@numba.jit('float64(float64, float64[:], int64)',
           nopython=True,
           cache=True,
           nogil=True)
def sum_of_vector_numba(result, x, sh):
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


def mult_row(x, row_vector):
    if row_vector.shape[0] != 1:
        row_vector = np.reshape(row_vector, (row_vector.shape[0], 1))
    elif row_vector.shape[0] != 1:
        row_vector = np.reshape(row_vector, (1, row_vector.shape[0]))
    if x.shape[0] != row_vector.shape[1]:
        row_vector = row_vector.T
    if x.shape[0] != row_vector.shape[1]:
        raise RuntimeError("Matrices haven't compatible size.")
    result = np.zeros(x.shape)
    mult_row_matrix_numba(row_vector.ravel(), x.toarray(), result, x.shape[0], x.shape[1])
    return result


@numba.jit('void(float64[:], float64[:,:], float64[:,:], int64, int64)',
           nopython=True,
           cache=True,
           nogil=True)
def mult_col_matrix_numba(column_vector, matrix, result, dim0, dim1):
    """
    Optimised matrix element-wise multiplication when one matrix
    :param column_vector:
    :param matrix:
    :param result:
    :param dim0:
    :param dim1:
    :return:
    """
    for i in range(0, dim0):
        for j in range(0, dim1):
            result[i, j] = column_vector[i] * matrix[i, j]


@numba.jit('void(float64[:], float64[:,:], float64[:,:], int64, int64)',
           nopython=True,
           nogil=True
           )
def mult_row_matrix_numba(row_vector, matrix, result, dim0, dim1):
    """
    Optimised matrix element-wise multiplication when one matrix
    :param row_vector:
    :param matrix:
    :param result:
    :param dim0:
    :param dim1:
    :return:
    """
    for i in range(0, dim0):
        for j in range(0, dim1):
            result[i, j] = row_vector[j] * matrix[i, j]


def mult_col(x, col_vector):
    if x.shape[0] == 1:
        x_newsize = np.reshape(x, (x.shape[1]))
    else:
        x_newsize = np.reshape(x, (x.shape[0]))
    xw_hat = np.zeros(col_vector.shape)
    mult_col_matrix_numba(x_newsize, col_vector, xw_hat, col_vector.shape[0], col_vector.shape[1])
    return xw_hat
