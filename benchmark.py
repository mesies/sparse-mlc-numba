import os

import numpy as np
from sklearn.linear_model import SGDClassifier

from MlcLinReg import MlcLinReg
from helpers import load_mlc_dataset, shuffle_dataset, split_train_test, tic, toc


def load_delicious():
    DATASET_FILENAME = 'delicious_data.txt'
    DATASET_TRAIN_SET_FILENAME = "delicious_trSplit.txt"
    DATASET_TEST_SET_FILENAME = "delicious_tstSplit.txt"

    DATASET_FILENAME = os.path.join('data', DATASET_FILENAME)
    DATASET_TRAIN_SET_FILENAME = os.path.join('data', DATASET_TRAIN_SET_FILENAME)
    DATASET_TEST_SET_FILENAME = os.path.join('data', DATASET_TEST_SET_FILENAME)

    X, y = load_mlc_dataset(DATASET_FILENAME,
                            header=True,
                            concatbias=True)
    f1 = open(DATASET_TRAIN_SET_FILENAME)
    train_ind = np.loadtxt(fname=f1, delimiter=" ", dtype=int)
    f1.close()

    f2 = open(DATASET_TEST_SET_FILENAME)
    test_ind = np.loadtxt(fname=f2, delimiter=" ", dtype=int)
    f2.close()

    # Normalize train and test indexes
    train_ind = train_ind - 1
    test_ind = test_ind - 1

    X_train = X[train_ind[:, 0]]
    X_test = X[test_ind[:, 0]]

    y_train = y[train_ind[:, 0]]
    y_test = y[test_ind[:, 0]]

    return X_train, y_train, X_test, y_test


from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load shuffle and split data
X_train, y_train, X_test, y_test = load_delicious()
X_train_s, y_train_s = shuffle_dataset(X_train, y_train)
X_train, y_train_f, X_test, y_test_f = split_train_test(X_train_s, y_train_s)

X_train2 = np.array(X_train.toarray())
X_test2 = np.array(X_test.toarray())
y_train2 = np.array(y_train_f[:, 4].toarray().ravel())
y_test2 = np.array(y_test_f[:, 4].toarray().ravel())

clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, verbose=0)
N = np.array(range(100, 1000, 10))
# N = np.array([50, 100, 200, 300, 400, 600])
# N = np.array([0.001, 0.005, 0.025, 0.05, 0.1])

scores = list()
scores_sgd = list()
times = list()
times_sgd = list()
feature = 61
batch_size = 8000
for n in N:
    t = tic()
    mlc = MlcLinReg(learning_rate=0.03, iterations=n, batch_size=batch_size, verbose=False, sparse=True)
    mlc.fit(X_train, y_train_f[:, feature])
    y_pred = mlc.predict(X_test)
    times.append(toc(t))
    scores.append(f1_score(y_pred=y_pred, y_true=y_test_f[:, feature].toarray()))

    t = tic()
    clf = SGDClassifier(loss='log', max_iter=200, tol=1e-3, verbose=0, shuffle=True)
    # clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train.toarray(), y_train_f[:, feature].toarray().ravel())
    y_pred2 = clf.predict(X_test.toarray())
    scores_sgd.append(f1_score(y_pred=y_pred2, y_true=y_test_f[:, feature].toarray()))
    times_sgd.append(toc(t))

plt.plot(N, times, label="Minibatch " + "batch size " + str(batch_size))
plt.plot(N, times_sgd, label="sklearn SGD")
plt.legend()
plt.show()
