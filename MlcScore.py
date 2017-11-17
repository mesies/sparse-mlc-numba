import numpy as np
from scipy import logical_and, logical_or


def score_accuracy(y_predicted, y_real):
    """
                (   y_real AND y_predicted  )
     score =    (  ------------------------ )
                (   y_real OR y_predicted   )

    :param y_predicted:
    :param y_real:
    :return:
    """

    if isinstance(y_predicted, np.ndarray):
        y_predicted = y_predicted.ravel()
    else:
        y_predicted = y_predicted.toarray().ravel()

    y_real = y_real.toarray().ravel()

    numerator = np.sum(1. * (logical_and(y_real, y_predicted)))
    denominator = np.sum(1. * (logical_or(y_real, y_predicted)))

    return numerator / denominator
