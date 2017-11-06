import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

import helpers
from MlcLinReg import MlcLinReg

(X, y) = sklearn.datasets.make_blobs(
    n_samples=1500,
    n_features=500,
    centers=2,
    cluster_std=1.05,
    random_state=20)

X = np.c_[np.ones((X.shape[0])), X]

np.float64(X)
np.float64(y)


mlc = MlcLinReg(learning_rate=0.001, iterations=2000, batch_size=200, verbose=True, sparse=False)
W = mlc.fit(X, y)
preds = mlc.predict(X)

print("Score " + str(accuracy_score(y_true=y, y_pred=preds)))

helpers.plot_linreg_results(X=X, W=W.w, y=y, preds=preds, lossHistory=mlc.lossHistory)

