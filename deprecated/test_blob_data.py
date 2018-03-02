import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from MlcLinReg import MlcLinReg
from helpers import *

(X, y) = sklearn.datasets.make_blobs(
    n_samples=1500,
    n_features=2,
    centers=2,
    cluster_std=2,
    random_state=20)

X = np.c_[np.ones((X.shape[0])), X]
X_train, y_train, X_test, y_test = split_train_test(X, y)

t = tic()

mlc = MlcLinReg(learning_rate=0.001, iterations=100, batch_size=300, verbose=False, sparse=True)
W = mlc.fit(sp.csr_matrix(X_train), sp.csr_matrix(y_train).T)
preds = mlc.predict(X_test)
print("Score " + str(accuracy_score(y_true=y_test, y_pred=preds)))

toc(t)

t = tic()
clf = RandomForestClassifier(n_estimators=200, n_jobs=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(type(clf))
print(f1_score(y_test, y_pred))
toc(t, str(type(clf)))
# helpers.plot_linreg_results(X=X, W=W.w, y=y, preds=preds, lossHistory=mlc.lossHistory)

from sklearn.linear_model import SGDClassifier

t = tic()
clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, verbose=0)
clf.fit(sp.csr_matrix(X_train), (y_train))
y_pred = clf.predict(X_test)
print(type(clf))
print(f1_score(y_test, y_pred))
toc(t, str(type(clf)))
