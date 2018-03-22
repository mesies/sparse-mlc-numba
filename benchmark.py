import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from MlcLinReg import MlcLinReg
from helpers import shuffle_dataset, split_train_test, tic, toc, load_delicious

"""
This script is used to compare performance of MlcLinReg and sklearn's SGDClassifier.
"""

# Load shuffle and split data
X_train, y_train, X_test, y_test = load_delicious()
X_train_s, y_train_s = shuffle_dataset(X_train, y_train)
X_train, y_train_f, X_test, y_test_f = split_train_test(X_train_s, y_train_s)

X_train2 = np.array(X_train.toarray())
X_test2 = np.array(X_test.toarray())
y_train2 = np.array(y_train_f[:, 4].toarray().ravel())
y_test2 = np.array(y_test_f[:, 4].toarray().ravel())

clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, verbose=0)
# N = np.array(range(100, 1000, 10))
# N = np.array([50, 100, 200, 300, 400, 600])
# N = np.array([0.001, 0.005, 0.025, 0.05, 0.1])
N = np.array([1000, 1500, 2000, 2500])
scores = list()
scores_sgd = list()
times = list()
times_sgd = list()
feature = 1
batch_size = 2048
iterations = 700

for n in N:
    t = tic()
    mlc1 = MlcLinReg(learning_rate=0.15, iterations=iterations, batch_size=n)
    mlc1.fit(X_train, y_train_f[:, feature])
    y_pred = mlc1.predict(X_test)
    times.append(toc(t))
    scores.append(f1_score(y_pred=y_pred, y_true=y_test_f[:, feature].toarray()))

    # t = tic()
    # mlc = MlcLinReg(learning_rate=0.01, iterations=iterations, batch_size=n, verbose=False, sparse=True,
    #                 regularization=0)
    # mlc.fit(X_train, y_train_f[:, feature])
    # y_pred = mlc.predict(X_test)
    # times_sgd.append(toc(t))
    # scores_sgd.append(f1_score(y_pred=y_pred, y_true=y_test_f[:, feature].toarray()))
    a = y_test_f[:, feature].toarray()
    t = tic()
    clf = SGDClassifier(loss='log', max_iter=n, tol=1e-3, verbose=0, shuffle=True, penalty="l1")
    # clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train.toarray(), y_train_f[:, feature].toarray().ravel())
    y_pred2 = clf.predict(X_test.toarray())
    scores_sgd.append(f1_score(y_pred=y_pred2, y_true=y_test_f[:, feature].toarray()))
    times_sgd.append(toc(t))


print scores
print times
plt.plot(N, scores, label="Minibatch " + "batch size " + str(batch_size))
plt.plot(N, scores_sgd, label="sklearn SGD")
plt.legend()
plt.show()

mlc1.plot_log_loss()
