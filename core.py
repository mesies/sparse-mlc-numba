from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from MlcLinReg import MlcLinReg
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
feature = 1
mlc = MlcLinReg(learning_rate=0.08, iterations=600, batch_size=2048,
                l_one=0.5)
mlc.fit(X_train, y_train[:, feature])
y_pred = mlc.predict(X_test)
print "ADAM linreg"
print(f1_score(y_pred=y_pred, y_true=y_test[:, feature].toarray()))

toc(ti)
t = tic()
clf = SGDClassifier(loss='log', max_iter=600, tol=1e-3)
# clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train[:, feature].toarray().ravel())
y_pred2 = clf.predict(X_test)
print "sklearn SGD"
print(f1_score(y_pred=y_pred2, y_true=y_test[:, feature].toarray()))
toc(t)
