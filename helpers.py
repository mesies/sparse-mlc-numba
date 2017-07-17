from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import numpy as np

"""
This file contains helper functions
"""


def load_mlc_dataset(
        filename,
        header=True):
    """
    This function extends 'load_svmlight_file' so that datasets that have a header
    are parsed correctly

    Args:
        filename: Path of the dataset.
        header: True if file has a header.

    Returns:
        X_train: csr_matrix(TODO check) which contains the features of each point.
        y_train: csr_matrix(TODO check) which contains the labels of each point.
        header_info: False if there is no header, contains an array which contains
                    0: Number of training examples
                    1: Feature Dimensionality
                    2: Label Dimensionality
    """
    f = open(filename)

    header_info = False
    if header:
        header_info = f.readline().split()
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,

        )
        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        # Convert y to sparse array
        ult = np.zeros((DATASET_SIZE, LABEL_NUMBER))
        for i in range(0, DATASET_SIZE):
            temp = np.zeros(LABEL_NUMBER)
            temp[np.asarray(y[i], dtype=int)] = 1
            ult[i] = temp
        y = csr_matrix(ult)
    else:
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,
        )
    f.close()

    return X, y, header_info
