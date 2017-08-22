import numpy as np
import scipy.sparse as sp
from scipy import logical_and, logical_or


def score_accuracy(ypredicted, yreal):
    """
    score = 1/N sum((y AND ypred)/(y OR ypred)))
    :param ypredicted:
    :param yreal:
    :return:
    """
    imax = yreal.shape[0]
    s = 0.
    for i in np.arange(0, imax):
        yr = (yreal[i, :].toarray())
        if isinstance(ypredicted[i, :], np.ndarray):
            yp = (ypredicted[i, :])
        else:
            yp = (ypredicted[i, :].toarray())
        ar = np.sum(1.*(logical_and(yr, yp)))
        pr = np.sum(1.*(logical_or(yr, yp)))
        if np.isnan(ar/pr):
            s += 0
        else:
            s += ar / pr
    s = s/imax
    return s


def score_test():
    y = sp.csr_matrix([[1, 1, 1], [1, 0, 0]])
    ypred = sp.csr_matrix([[1, 1, 1], [1, 1, 0]])
    print score_accuracy(ypred, y)