import numba
import numpy as np
from numba import prange
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

"""
This file implements low-level functions written utilising the numba framework
,in order to improve speed when input is sparse.
"""


def coo_dot(A, B):
    # A is M x N, coo matrix
    # B is N x 1

    results = np.zeros((A.shape[0], 1))
    spmv(A.data, A.row, A.col, B, results)

    assert results.shape == A.dot(B.T).shape
    return results


def spmv(vals, rows, cols, vec, results):
    for i in range(rows.size - 1):
        # slice the corresponding column coordinates
        # and values for each row
        col_coords = cols[rows[i]:rows[i + 1]]
        data = vals[rows[i]:rows[i + 1]]

        # dot product with the vector
        result = 0
        for col_coord, datum in zip(col_coords, data):
            result += (vec[col_coord] * datum)

        results[i] = result


@numba.jit('void(float64[:,:], float64[:], int32[:], int32[:], int64)',
           nopython=True
           )
def mult_row_sparse_numba(result, matrix, result_row, result_col, lenrow):
    for i in xrange(0, lenrow):
        row = np.take(result_row, i)
        col = np.take(result_col, i)
        result[row, col] *= np.take(matrix, col)


def nonzero(x):
    """
    Returns indices of non zero elements of a scipy csr_matrix, wrapper for nonzero_numb.
    @param x: The sparse matrix in question only csr and csc matrices are supported
    @return: row indices and column indices
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
            'void(int32[:], int32[:], int32[:], int32[:], int64, int64)'
            ],
           nopython=True,
           nogil=True,
           cache=True,
           fastmath=True)
def nonzero_numba(result_row, result_col, indices, indptr, x_shape_0, iscsr):
    """
    See nonzero
    @param result_row:
    @param result_col:
    @param indices:
    @param indptr:
    @param x_shape_0:
    @param iscsr:
    @return:
    """
    # column indices for column i is in indices[indptr[i]:indptr[i+1]]
    if iscsr == 1:
        h = 0
        for i in range(0, x_shape_0):
            ind = indices[indptr[i]:indptr[i + 1]]
            for j in ind:
                result_row[h] = i
                result_col[h] = j
                h += 1
    else:
        h = 0
        for i in range(0, x_shape_0):
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
    Numba function which returns sum of eaxh row
     e.g. [1 2 3] -> 6
     [1 1]    2
     [2 2] -> 4
     [3 3]    6
    @param x:
    @param result:
    @param dim0:
    @param dim1:
    @return:
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
    @param result:
    @param x:
    @param sh:
    @return:
    """
    s = x[0]
    for i in range(1, sh):
        s = s + x[i]
    result = -s
    return result


def mult_col_raw_col_row_sum_raw(x, row_vector):
    """
    A combination of @col_row_sum_raw and @mult_mult_row_raw functions in order to save some
    memory and improve performance.
    @param x:
    @param row_vector:
    @return:
    """
    # on deliciousLarge -> 8.5s per iter BEFORE
    #

    result_data = x.data.copy()
    result_row = np.zeros(x.data.shape[0], dtype=int)
    result_col = np.zeros(x.data.shape[0], dtype=int)

    result = np.zeros(x.shape[1])

    # data, result_row, result_col = mult_row_raw(x, row_vector)
    # result = col_row_sum_raw(data, result_row, result_col, x.shape[0], x.shape[1])
    #
    mult_col_raw_col_row_sum_raw_numba(
        result,
        row_vector.ravel(),
        x.indptr,
        x.indices,
        result_data,
        result_row,
        result_col,
        x.shape[0],
        x.shape[1])

    return np.reshape(result, (result.shape[0], 1))


@numba.jit(['void(float64[:], float64[:], int32[:], int32[:], float64[:], int32[:], int32[:], int64, int64)',
            'void(float64[:], float64[:], int32[:], int32[:], float64[:], int64[:], int64[:], int64, int64)'],
           nopython=True,
           nogil=True,
           fastmath=True
           )
def mult_col_raw_col_row_sum_raw_numba(result,
                                       row_vector,
                                       x_indptr,
                                       x_indices,
                                       result_data,
                                       result_row,
                                       result_col,
                                       x_dim0,
                                       x_dim1):
    # mult row raw
    h = 0

    for i in range(0, x_dim0):

        result_data[x_indptr[i]:x_indptr[i + 1]] *= row_vector[i]
        col_ind = x_indices[x_indptr[i]:x_indptr[i + 1]]
        for k in col_ind:
            result_col[h] = k
            result_row[h] = i
            h += 1

            # result[result_col[h]] += result_data[h]
    for j in prange(0, len(result_data)):
        result[result_col[j]] += result_data[j]


# @profile
def col_row_sum_raw(data, row, col, shape0, shape1):
    """
    Has same function as coo_row_sum but its input is, instead of
    a single coo_matrix, its components.
    @param data:
    @param row:
    @param col:
    @param shape0:
    @param shape1:
    @return:
    """
    result = np.zeros(shape1)
    col_row_sum_numba(result, data, row, col, shape0, shape1, data.shape[0])
    return np.reshape(result, (1, result.shape[0]))


def mult_row_raw(x, row_vector):
    """
    Has same function as mult_raw but returns rather than a coo_matrix
    , returns its components in order to improve speed
    @param x:
    @param row_vector:
    @return:
    """
    result_data = x.data.copy()
    result_row = np.zeros(x.data.shape[0], dtype=int)
    result_col = np.zeros(x.data.shape[0], dtype=int)

    mult_row_matrix_numba(row_vector.ravel(), x.indptr, x.indices, result_data, result_row, result_col, x.shape[0],
                          x.shape[1])

    return result_data, result_row, result_col


def mult_row(x, row_vector):
    """
    Multiplies each column of matrix x ,element wise , by a column vector.
    e.g.
    1 2 3       1       1 2 3
    1 2 3   x   2  ->   2 4 6
    1 2 3       3       3 6 9
    1 2 3       4       4 8 12

    @param x:
    @param row_vector:
    @return:
    """
    result_data = x.data.copy()
    result_row = np.zeros(x.data.shape[0], dtype=int)
    result_col = np.zeros(x.data.shape[0], dtype=int)

    mult_row_matrix_numba(row_vector.ravel(), x.indptr, x.indices, result_data, result_row, result_col, x.shape[0],
                          x.shape[1])

    # We must return coo
    # result = csr_matrix((result_data, x.indices, x.indptr), shape=x.shape)
    # assert result.shape == x.shape

    return coo_matrix((result_data, (result_row, result_col)), shape=x.shape)


@numba.jit(['void(float64[:], int32[:], int32[:], float64[:], int32[:], int32[:], int64, int64)',
            'void(float64[:], int32[:], int32[:], float64[:], int64[:], int64[:], int64, int64)'],
           nopython=True,
           nogil=True,
           fastmath=True
           )
def mult_row_matrix_numba(row_vector, x_indptr, x_indices, result_data, result_row, result_col, x_dim0, x_dim1):
    h = 0

    for i in range(0, x_dim0):
        result_data[x_indptr[i]:x_indptr[i + 1]] *= row_vector[i]
        col_ind = x_indices[x_indptr[i]:x_indptr[i + 1]]
        for k in col_ind:
            result_col[h] = k
            result_row[h] = i
            h += 1


@numba.jit(['void(float64[:], float64[:,:], float64[:,:], int64, int64)'],
           nopython=True,
           cache=True,
           nogil=True)
def mult_col_matrix_numba(column_vector_1d, matrix_2d, result_2d, dim0, dim1):
    """
    Optimised matrix element-wise multiplication when one matrix
    @param column_vector_1d:
    @param matrix_2d:
    @param result_2d:
    @param dim0:
    @param dim1:
    @return:
    """
    for i in range(0, dim0):
        for j in range(0, dim1):
            result_2d[i, j] = column_vector_1d[i] * matrix_2d[i, j]


def mult_col(col_vector_1d, matrix_2d):
    """

    @param col_vector_1d:
    @param matrix_2d:
    @return:
    """
    if col_vector_1d.shape[0] == 1:
        col_vector_1d_new_shape = np.reshape(col_vector_1d, (col_vector_1d.shape[1]))
    else:
        col_vector_1d_new_shape = np.reshape(col_vector_1d, (col_vector_1d.shape[0]))

    result_2d = np.zeros(matrix_2d.shape)
    mult_col_matrix_numba(col_vector_1d_new_shape, matrix_2d, result_2d, matrix_2d.shape[0], matrix_2d.shape[1])
    return result_2d


def coo_row_sum(A):
    """
    Function which returns sum of eaxh row,
     e.g. [1 2 3] -> 6
     [1 1]    2
     [2 2] -> 4
     [3 3]    6
    @param A:
    @return:
    """
    if not isinstance(A, coo_matrix):
        raise NotImplementedError
    result = np.zeros((A.shape[1]))
    col_row_sum_numba(result, A.data, A.row, A.col, A.shape[0], A.shape[1], A.data.shape[0])
    return np.reshape(result, (1, result.shape[0]))


# Parallel??
@numba.jit(['void(float64[:], float64[:], int32[:], int32[:], int64, int64, int64)',
            'void(float64[:], float64[:], int64[:], int64[:], int64, int64, int64)'],
           nopython=True,
           nogil=True,
           fastmath=True)
def col_row_sum_numba(result, data, row_ind, col_ind, row_no, col_no, data_no):
    for k in range(0, data_no):
        result[col_ind[k]] += data[k]
