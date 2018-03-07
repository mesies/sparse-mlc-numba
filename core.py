from MlcClassifierChains import MlcClassifierChains
from MlcScore import score_accuracy
from helpers import load_mlc_dataset, tic, toc, save_sparse_csr, load_sparse_csr

ti = tic()
# DATASET_TRAIN_SET_FILENAME = "data\\deliciousLarge_train.txt"
# DATASET_TEST_SET_FILENAME = "data\\deliciousLarge_test.txt"

# DATASET_TRAIN_SET_FILENAME = "data\\rcv1x_train.txt"
# DATASET_TEST_SET_FILENAME = "data\\rcv1x_test.txt"

DATASET_TRAIN_SET_FILENAME = "data\\eurlex_train.txt"
DATASET_TEST_SET_FILENAME = "data\\eurlex_test.txt"

# DATASET_TRAIN_SET_FILENAME = "data\\amazonCat_train.txt"
# DATASET_TEST_SET_FILENAME = "data\\amazonCat_test.txt"

# Requires at least 12 GB RAM
# DATASET_TEST_SET_FILENAME = "data\\amazonCat-14K_test.txt"
# DATASET_TRAIN_SET_FILENAME = "data\\amazonCat-14K_train.txt"

# DATASET_TRAIN_SET_FILENAME = "data\\amazon-3M_train.txt"
# DATASET_TEST_SET_FILENAME = "data\\amazon-3M_test.txt"

# DATASET_TRAIN_SET_FILENAME = "data\\wiki10_train.txt"
# DATASET_TEST_SET_FILENAME = "data\\wiki10_test.txt"
try:
    print("Attempting Load from local files")
    X_train = (load_sparse_csr("xtrain_big.npz"))
    X_test = (load_sparse_csr("xtest_big.npz"))
    y_train = (load_sparse_csr("ytrain_big.npz"))
    y_test = (load_sparse_csr("ytest_big.npz"))
    print("Loading Completed")
except IOError:
    print("Loading Failed")
    print("Started loading from dataset")
    X_train, y_train = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                        header=True,
                                        concatbias=True)

    X_test, y_test = load_mlc_dataset(DATASET_TRAIN_SET_FILENAME,
                                      header=True,
                                      concatbias=True)
    save_sparse_csr("xtrain_big", X_train)
    save_sparse_csr("xtest_big", X_test)
    save_sparse_csr("ytrain_big", y_train)
    save_sparse_csr("ytest_big", y_test)

print("Started Fitting")
mlc = MlcClassifierChains(learning_rate=0.01,
                          iterations=100,
                          batch_size=100,
                          alpha=0.99,
                          velocity=1)
mlc.fit(X_train, y_train)
y_pred = mlc.predict(X_test)

print(score_accuracy(y_pred, y_test))

toc(ti)
