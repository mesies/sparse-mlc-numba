import numba
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix


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


def mult_row_raw(x, row_vector):
    result_data = x.data.copy()
    result_row = np.zeros(x.data.shape[0], dtype=int)
    result_col = np.zeros(x.data.shape[0], dtype=int)

    mult_row_matrix_numba(row_vector.ravel(),
                          x.indptr,
                          x.indices,
                          result_data,
                          result_row,
                          result_col,
                          x.shape[0],
                          x.shape[1])

    # We must return coo
    # result = csr_matrix((result_data, x.indices, x.indptr), shape=x.shape)
    # assert result.shape == x.shape

    return result_data, result_row, result_col


def mult_row(x, row_vector):
    result_data = x.data.copy()
    result_row = np.zeros(x.data.shape[0], dtype=int)
    result_col = np.zeros(x.data.shape[0], dtype=int)

    mult_row_matrix_numba(row_vector.ravel(),
                          x.indptr,
                          x.indices,
                          result_data,
                          result_row,
                          result_col,
                          x.shape[0],
                          x.shape[1])

    # We must return coo
    # result = csr_matrix((result_data, x.indices, x.indptr), shape=x.shape)
    # assert result.shape == x.shape

    return coo_matrix((result_data, (result_row, result_col)), shape=x.shape)


@numba.jit('void(float64[:], int32[:], int32[:], float64[:], int32[:], int32[:], int64, int64)',
           nopython=True,
           nogil=True
           )
def mult_row_matrix_numba(row_vector,
                          x_indptr,
                          x_indices,
                          result_data,
                          result_row,
                          result_col,
                          dim0,
                          dim1):
    # i -> row
    h = 0

    for i in range(0, dim0):
        result_data[x_indptr[i]:x_indptr[i + 1]] *= row_vector[i]
        col_ind = x_indices[x_indptr[i]:x_indptr[i + 1]]
        for k in col_ind:
            result_col[h] = (k)
            result_row[h] = (i)
            h += 1

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





def mult_col(x, col_vector):
    if x.shape[0] == 1:
        x_newsize = np.reshape(x, (x.shape[1]))
    else:
        x_newsize = np.reshape(x, (x.shape[0]))
    xw_hat = np.zeros(col_vector.shape)
    mult_col_matrix_numba(x_newsize, col_vector, xw_hat, col_vector.shape[0], col_vector.shape[1])
    return xw_hat


def col_row_sum_raw(data, row, col, shape0, shape1):
    result = np.zeros((shape1))
    col_row_sum_numba(result, data, row, col, shape0, shape1, data.shape[0])
    return np.reshape(result, (1, result.shape[0]))

def coo_row_sum(A):
    if not isinstance(A, coo_matrix):
        raise NotImplementedError
    result = np.zeros((A.shape[1]))
    col_row_sum_numba(result, A.data, A.row, A.col, A.shape[0], A.shape[1], A.data.shape[0])
    return np.reshape(result, (1, result.shape[0]))


# Parallel??
@numba.jit('void(float64[:], float64[:], int32[:], int32[:], int64, int64, int64)',
           nopython=True,
           nogil=True)
def col_row_sum_numba(result, data, row_ind, col_ind, row_no, col_no, data_no):
    for k in range(0, data_no):
        result[col_ind[k]] += data[k]
