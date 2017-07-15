from sklearn.datasets import load_svmlight_file

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

    X_train, y_train = load_svmlight_file(
        f=f,
        multilabel=True
    )
    f.close()

    return X_train, y_train, header_info
