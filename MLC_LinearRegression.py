import math

import numpy as np

"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MLC_LinearRegression:
    def __init__(self, X_train, y_train, l):
        self.X_train = X_train
        self.y_train = y_train
        self.l = l
        self.w = np.array([0, 0])

    def fit(self, X, y):

        ITERATIONS = 1000

        self.w = self.gradient_decent(self, X, y, self.w, ITERATIONS)

        # Simple Gradient Descent

    def gradient_decent(self, X, y, w, tol=10 ^ (-3), iterations=10000):

        L_old = np.inf

        for i in range(0, iterations):
            wt_x = np.dot(np.transpose(self.w), X)
            sigm_wt_x = self.sigm(wt_x)

            L = - np.sum(
                y * np.log(sigm_wt_x) + (1 - y) * np.log(1 - sigm_wt_x))

            if np.abs(L - L_old) < tol:
                break

            self.w = self.w + self.l * (
                np.dot((-(np.transpose(X))), y)
                + np.dot(
                    np.dot(
                        np.transpose(X), X), self.w)
            )
        return w

    def predict(self, X):
        y = self.sigm(np.dot(np.transpose(self.w), X))
        y = np.around(y, decimals=-1)
        return y

    def sigm(x):
        if np.sign(x) >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            return math.exp(x) / (1 + math.exp(x))
