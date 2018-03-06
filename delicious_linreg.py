import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from MlcLinReg import MlcLinReg
from helpers import load_mlc_dataset, tic, toc

ti = tic()
DATASET_FILENAME = "data\delicious_data.txt"
DATASET_TRAIN_SET_FILENAME = "data\delicious_trSplit.txt"
DATASET_TEST_SET_FILENAME = "data\delicious_tstSplit.txt"

#   Debug Options
DEBUG = 0
DEBUG_DATASET_SIZE = 50

# 1. Load dataset
#   1.1 Load data from delicious dataset, also use sklearn's sparse data structures.
X, y = load_mlc_dataset(DATASET_FILENAME,
                        header=True,
                        concatbias=True)


#   1.2 Split into train/test sets according to delicious_trSplit.txt and delicious_tstSplit.txt, both files
#       contain indexes.

f1 = open(DATASET_TRAIN_SET_FILENAME)
train_ind = np.loadtxt(fname=f1, delimiter=" ", dtype=int)
f1.close()

f2 = open(DATASET_TEST_SET_FILENAME)
test_ind = np.loadtxt(fname=f2, delimiter=" ", dtype=int)
f2.close()

# Normalize train and test indexes
train_ind = train_ind - 1
test_ind = test_ind - 1

if DEBUG == 1:
    X_train = X[train_ind[:(DEBUG_DATASET_SIZE * 4), 0]]
    X_test = X[test_ind[:DEBUG_DATASET_SIZE, 0]]

    y_train = y[train_ind[:DEBUG_DATASET_SIZE * 4, 0]]
    y_test = y[test_ind[:DEBUG_DATASET_SIZE, 0]]
else:
    X_train = X[train_ind[:, 0]]
    X_test = X[test_ind[:, 0]]

    y_train = y[train_ind[:, 0]]
    y_test = y[test_ind[:, 0]]

y_train = y_train[:, 0]
y_test = y_test[:, 0]

# 2. Implement Binary Logistic Regression Classifier
#   2.1 Code likelihood
#   2.2 Code Stochastic Gradient Descent with regards to dataset sparsity
#
# Notes : Code it like sklearn classifiers
mlc = MlcLinReg(learning_rate=0.09,
                iterations=1000,
                batch_size=5,
                velocity=0.9)
mlc.fit(X_train, y_train)
y_pred = mlc.predict(X_test)

print("Score " + str(accuracy_score(y_true=y_test.toarray(), y_pred=y_pred)))
toc(ti)
fig = plt.figure()

plt.plot(np.arange(0, len(mlc.lossHistory)), mlc.lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
# 3. Implement Classifier Chains
#   3.1 Code classifier chains with regards to interchangeable classifiers objects

# 4. Implement Scoring Function
#   4.1 Code Scoring function as described in report.pdf
