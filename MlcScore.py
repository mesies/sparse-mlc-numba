import numpy as np
from hyperdash.sdk import monitor
from scipy import logical_and, logical_or


def score_accuracy(ypredicted, yreal):
    """
    score = 1/N sum((y AND ypred)/(y OR ypred)))
    :param ypredicted:
    :param yreal:
    :return:
    """

    s = 0.
    if isinstance(ypredicted, np.ndarray):
        yp = (ypredicted)
    else:
        yp = (ypredicted.toarray())

    yp = yp.ravel()
    yr = yreal.toarray().ravel()
    ar = np.sum(1. * (logical_and(yr, yp)))
    pr = np.sum(1. * (logical_or(yr, yp)))
    s = ar / pr

    return(s)
