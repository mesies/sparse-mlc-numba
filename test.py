from sklearn.metrics import accuracy_score

import helpers
import sklearn
import logging
import numpy as np
from MLC_LinearRegression import MLC_LinearRegression
import matplotlib.pyplot as plt

(X, y) = sklearn.datasets.make_blobs(
    n_samples=400,
    n_features=70,
    centers=2,
    cluster_std=1.05,
    random_state=20)

X = np.c_[np.ones((X.shape[0])), X]

np.float64(X)
np.float64(y)


mlc = MLC_LinearRegression(learning_rate=0.001)
W = mlc.fit(X, y)
preds = mlc.predict(X)

print("Score " + str(accuracy_score(y_true=y, y_pred=preds)))

#helpers.plot_linreg_results(X, W, y, preds, mlc.lossHistory)

