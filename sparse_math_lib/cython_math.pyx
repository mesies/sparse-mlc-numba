cimport cython
cimport numpy as np

# broken

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
cpdef void mult_cython(np.ndarray[double, ndim=2] A,
                       np.ndarray[double, ndim=2] B,
                       np.ndarray[int, ndim=1] Annzcol,
                       np.ndarray[int, ndim=1] Annzrow,
                       int lengh):
    cdef int i, row, col

    with nogil:
        for i in range(lengh):
            row = Annzrow[i]
            col = Annzcol[i]
            A[row, col] *= B[col, 0]
