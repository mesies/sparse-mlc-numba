import numpy as np
from helpers import load_mlc_dataset, tic, toc
from MlcLinReg import MlcLinReg
from MlcClassifierChains import MlcClassifierChains
from helpers import save_sparse_csr, load_sparse_csr

ti = tic()
DATASET_TRAIN_SET_FILENAME = "data\\rcv1x_train.txt"
DATASET_TEST_SET_FILENAME = "data\\rcv1x_test.txt"

DEBUG = 0
DEBUG_DATASET_SIZE = 50

try:
    print "Attempting Load from local files"
    X_train = (load_sparse_csr("xtrain.npz"))
    X_test = (load_sparse_csr("xtest.npz"))
    y_train = (load_sparse_csr("ytrain.npz"))
    y_test = (load_sparse_csr("ytest.npz"))
except IOError:
    print "Loading Failed"
    print "Started loading from dataset"
    X_train, y_train, header_info_train = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                                           header=True,
                                                           concatbias=True)

    X_test, y_test, header_info_train = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                                         header=True,
                                                         concatbias=True)
    save_sparse_csr("xtrain", X_train)
    save_sparse_csr("xtest", X_test)
    save_sparse_csr("ytrain", y_train)
    save_sparse_csr("ytest", y_test)
print "Started Fitting"
mlc = MlcClassifierChains(MlcLinReg,
                          learning_rate=0.01,
                          iterations=10,
                          sparse=True,
                          verbose=False,
                          grad_check=False,
                          batch_size=25,
                          alpha=0.5,
                          velocity=1)
mlc.fit(X_train, y_train)
ypred = mlc.predict(X_test)

from MlcScore import score_accuracy
print score_accuracy(ypred, y_test)

toc(ti)