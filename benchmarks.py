import sys

import numpy as np
from helpers import load_mlc_dataset, tic, toc
from MlcLinReg import MlcLinReg
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from MlcClassifierChains import MlcClassifierChains

ti = tic()
DATASET_FILENAME = "data\delicious_data.txt"
DATASET_TRAIN_SET_FILENAME = "data\delicious_trSplit.txt"
DATASET_TEST_SET_FILENAME = "data\delicious_tstSplit.txt"

#   Debug Options
DEBUG = 0
DEBUG_DATASET_SIZE = 50

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

mlc = MlcClassifierChains(MlcLinReg,
                          learning_rate=0.001,
                          iterations=100,
                          sparse=True,
                          verbose=False,
                          grad_check=False,
                          batch_size=20,
                          alpha=0.5,
                          velocity=1)
mlc.fit(X_train, y_train)
mlc.predict(X_test)

toc(ti)