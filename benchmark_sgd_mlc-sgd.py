import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from MlcLinReg import MlcLinReg
from helpers import shuffle_dataset, split_train_test, tic, toc, load_delicious

"""
This script is used to compare performance of MlcLinReg and sklearn's SGDClassifier.
"""
feature = 1

# Load shuffle and split data
X_train, y_train, X_test, y_test = load_delicious(feature)

X_train_s, y_train_s = shuffle_dataset(X_train, y_train)

X_train, y_train, X_test, y_test = split_train_test(X_train_s, y_train_s)

# N = np.array(range(100, 1000, 10))
# N = np.array([50, 100, 200, 300, 400, 600])
# N = np.array([0.001, 0.005, 0.025, 0.05, 0.1])
N = np.array([5, 32, 64, 100, 128, 200, 256, 350, 512, 700, 1024, 1500, 2048])

scores = list()
scores_sgd = list()
times = list()
times_sgd = list()

batch_size = 2048
iterations = 200

for n in N:
    mlc1 = MlcLinReg(learning_rate=0.005, iterations=iterations, batch_size=n, optimiser="default")

    t = tic()
    # mlc1.bayesianCVparams(X_train, y_train_f[:, feature])
    X_train, y_train = shuffle_dataset(X_train, y_train, True)

    mlc1.fit(X_train, y_train)
    print toc(t, True)
    times.append(toc(t))
    y_pred = mlc1.predict(X_test)

    times.append(toc(t))
    scores.append(f1_score(y_pred=y_pred, y_true=y_test.toarray()))
    #mlc1.plot_log_loss()
    # t = tic()
    # mlc = MlcLinReg(learning_rate=0.01, iterations=iterations, batch_size=n, verbose=False, sparse=True,
    #                 regularization=0)
    # mlc.fit(X_train, y_train_f[:, feature])
    # y_pred = mlc.predict(X_test)
    # times_sgd.append(toc(t))
    # scores_sgd.append(f1_score(y_pred=y_pred, y_true=y_test_f[:, feature].toarray()))

    X_train, y_train = shuffle_dataset(X_train, y_train, True)

    t = tic()
    clf = SGDClassifier(loss='log', max_iter=iterations, tol=1e-3, verbose=0, shuffle=True)
    clf.fit(X_train.toarray(), y_train.toarray().ravel())
    y_pred2 = clf.predict(X_test.toarray())
    scores_sgd.append(f1_score(y_pred=y_pred2, y_true=y_test.toarray()))
    times_sgd.append(toc(t))

print "mlc-sgd"
print times
print "sklearn sgd"
print times_sgd

plt.plot(N, scores, label="MLC-SGD Mini-Batch " + " Score")
plt.plot(N, scores_sgd, label="sklearn SGD" + " Score")
plt.xlabel('MLC-SGD Batch Size')
plt.ylabel('F1 score')
plt.legend()
plt.show()
