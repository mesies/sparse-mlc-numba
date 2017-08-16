import scipy.sparse as sp
import numpy as np
from mathutil import nonzero
from scipy.sparse.sparsetools import csr_tocsc, csr_tobsr, csr_count_blocks, \
        get_csr_submatrix, csr_sample_values

def _get_submatrix(x, row_slice, col_slice):
        """
        Return a submatrix of this matrix (new matrix is created).
        Modified to improve performance, origina version scipy version 0.19.1
        """

        M,N = x.shape

        def process_slice(sl, num):
            #if isinstance(sl, slice): #Omitable
            # if sl.step not in (1, None):#Omitable
            # raise ValueError('slicing with step != 1 not supported')#Omitable
            i0, i1 = sl.start, sl.stop
            if i0 is None:
                i0 = 0
            elif i0 < 0:
                i0 = num + i0

            if i1 is None:
                i1 = num
            elif i1 < 0:
                i1 = num + i1
            return i0, i1

            # if not isinstance(sl, slice) & sp.csr.isintlike(sl):
            #     if sl < 0:
            #         sl += num
            #     return sl, sl + 1
            # #else:#Omitable
            #     #raise TypeError('expected slice or scalar')#Omitable

        def check_bounds(i0, i1, num):#Omitable
            if not (0 <= i0 <= num) or not (0 <= i1 <= num) or not (i0 <= i1):#Omitable
                raise IndexError(#Omitable
                      "index out of bounds: 0 <= %d <= %d, 0 <= %d <= %d,"#Omitable
                      " %d <= %d" % (i0, num, i1, num, i0, i1))#Omitable

        i0, i1 = process_slice(row_slice, M)
        j0, j1 = process_slice(col_slice, N)
        #check_bounds(i0, i1, M)#Omitable
        #check_bounds(j0, j1, N)#Omitable

        indptr, indices, data = get_csr_submatrix(M, N,
                x.indptr, x.indices, x.data,
                int(i0), int(i1), int(j0), int(j1))

        shape = (i1 - i0, j1 - j0)

        return sp.csr_matrix((data,indices,indptr), shape=shape)

DATASET_FILENAME = "data\delicious_data.txt"
DATASET_TRAIN_SET_FILENAME = "data\delicious_trSplit.txt"
DATASET_TEST_SET_FILENAME = "data\delicious_tstSplit.txt"

#   Debug Options
DEBUG = 0
DEBUG_DATASET_SIZE = 50

from helpers import load_mlc_dataset
X, y, header_info = load_mlc_dataset(DATASET_FILENAME,
                                     header=True,
                                     concatbias=True)
DATASET_SIZE = int(header_info[0])
FEATURE_NUMBER = int(header_info[1])
LABEL_NUMBER = int(header_info[2])

#   1.2 Split into train/test sets according to delicious_trSplit.txt and delicious_tstSplit.txt, both files
#       contain indexes.

f1 = open(DATASET_TRAIN_SET_FILENAME)
train_ind = np.loadtxt(fname=f1, delimiter=" ", dtype=int)
f1.close()

f2 = open(DATASET_TEST_SET_FILENAME)
test_ind = np.loadtxt(fname=f2, delimiter=" ", dtype=int)
f2.close()

# Normalize train and test indexes
train_ind = train_ind - 1
test_ind = test_ind - 1

if DEBUG == 1:
    X_train = X[train_ind[:(DEBUG_DATASET_SIZE * 4), 0]]
    X_test = X[test_ind[:DEBUG_DATASET_SIZE, 0]]

    y_train = y[train_ind[:DEBUG_DATASET_SIZE * 4, 0]]
    y_test = y[test_ind[:DEBUG_DATASET_SIZE, 0]]
else:
    X_train = X[train_ind[:, 0]]
    X_test = X[test_ind[:, 0]]

    y_train = y[train_ind[:, 0]]
    y_test = y[test_ind[:, 0]]

y_train = y_train[:, 0]
y_test = y_test[:, 0]


X = X_train

A = sp.csr_matrix(np.array(
    [   #0  1  2  3
        [1, 0, 1, 0],#0
        [0, 0, 0, 0],#1
        [0, 0, 1, 0],#2
        [1, 1, 1, 0] #3
    ]))
from helpers import tic, toc
testrange = 10000
t= tic()
for i in range(testrange):
    sub = _get_submatrix(X, slice(None, testrange), slice(X.shape[1]))
toc(t, "Custom function")
t= tic()
for i in range(testrange):
    sub1 = X[0:testrange,:]
toc(t, "duiltin")

assert sub.nnz == sub1.nnz
result = []
resultrow = np.zeros((A.data.shape[0]))
resultcol = np.zeros((A.data.shape[0]))


def next_batch(self, X, y, batchSize):
    for i in np.arange(0, X.shape[0], int(batchSize)):
        limit = (i + batchSize)
        if limit > X.shape[0]: limit = X.shape[0]
        if sp.issparse(X):
            # 18/41 sec of execution
            yield (X[i:limit, :], y[i:limit, :])
        else:
            yield (X[i:limit, :], y[i:limit])


