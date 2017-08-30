import numpy as np
import scipy.sparse as sp
from MlcClassifierChains import MlcClassifierChains
from MlcLinReg import MlcLinReg
from helpers import load_mlc_dataset, tic, toc
from MlcScore import score_accuracy

# Comment when debuging with line profiler
profile = lambda f: f

ti = tic()
DATASET_FILENAME = "data\delicious_data.txt"
DATASET_TRAIN_SET_FILENAME = "data\delicious_trSplit.txt"
DATASET_TEST_SET_FILENAME = "data\delicious_tstSplit.txt"


# 1. Load dataset
#   1.1 Load data from delicious dataset, also use sklearn's sparse data structures.
X, y, header_info = load_mlc_dataset(DATASET_FILENAME,
                                     header=True,
                                     concatbias=True)
DATASET_SIZE = int(header_info[0])
FEATURE_NUMBER = int(header_info[1])
LABEL_NUMBER = int(header_info[2])

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

X_train = X[train_ind[:, 0]]
X_test = X[test_ind[:, 0]]

y_train = y[train_ind[:, 0]]
y_test = y[test_ind[:, 0]]

mlc = MlcClassifierChains(MlcLinReg,
                          learning_rate=0.001,
                          iterations=100,
                          sparse=True,
                          verbose=False,
                          grad_check=False,
                          batch_size=128,
                          alpha=0.5,
                          velocity=0.9)

mlc.fit(X_train, y_train)

y_pred = mlc.predict(X_test)

print(score_accuracy(y_pred, y_test))

toc(ti)
