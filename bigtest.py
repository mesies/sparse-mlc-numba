import sys
import numpy as np
from helpers import load_mlc_dataset, tic, toc
from MlcLinReg import MlcLinReg
from MlcClassifierChains import MlcClassifierChains

sys.setcheckinterval(250)

ti = tic()
DATASET_TRAIN_SET_FILENAME = "data\\rcv1x_train.txt"
DATASET_TEST_SET_FILENAME = "data\\rcv1x_test.txt"

DEBUG = 0
DEBUG_DATASET_SIZE = 50

X_train, y_train, header_info_train = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                     header=True,
                                     concatbias=True)

X_test, y_test, header_info_train = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                     header=True,
                                     concatbias=True)


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