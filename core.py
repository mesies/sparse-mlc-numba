import sklearn as sk
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from helpers import load_mlc_dataset

DATASET_FILENAME = "delicious_data.txt"
DATASET_TRAIN_SET_FILENAME = "delicious_trSplit.txt"
DATASET_TEST_SET_FILENAME = "delicious_tstSplit.txt"

# 1. Load dataset
#   1.1 Load data from delicious dataset, also use sklearn's sparse data structures.
X, y, header_info = load_mlc_dataset(DATASET_FILENAME,
                                     header=True)
#   1.2 Split into train/test sets according to delicious_trSplit.txt and delicious_tstSplit.txt, both files
#       contain indexes.

f1 = open(DATASET_TRAIN_SET_FILENAME)
train_ind = np.loadtxt(fname=f1, delimiter=" ")
f1.close()

f2 = open(DATASET_TEST_SET_FILENAME)
test_ind = np.loadtxt(fname=f2, delimiter=" ")
f2.close()

X_train = X[train_ind[:, 0]]
X_test = X[test_ind[:, 0]]

y_train = y[train_ind[:, 0]]
y_test = y[test_ind[:, 0]]


# 2. Implement Binary Logistic Regression Classifier
#   2.1 Code likelihood
#   2.2 Code Stochastic Gradient Descent with regards to dataset sparsity
#
# Notes : Code it like sklearn classifiers

# 3. Implement Classifier Chains
#   3.1 Code classifier chains with regards to interchangeable classifiers objects

# 4. Implement Scoring Function
#   4.1 Code Scoring function as described in report.pdf