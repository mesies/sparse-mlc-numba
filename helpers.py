import itertools
import logging
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from scipy import interp
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import train_test_split as _sklearn_split

from sparse_math_lib.gradient import gradient_sp
from sparse_math_lib.logloss import log_likelihood_sp, log_likelihood

# Comment when debugging with line_profiler
profile = lambda f: f

"""
This file contains helper functions
"""


def split_train_test(X, y, train_ratio=0.8):
    """
    Splits X, y into a train and a test set. Returns X_train, y_train, X_test, y_test
    @param X
    @param y
    @param train_ratio
    @return Split dataset
    """
    training_size = X.shape[0]
    indices = list(range(training_size))
    tr_indices = indices[:(int(len(indices) * train_ratio))]
    ts_indices = indices[int(len(indices) * train_ratio) - 1:]

    X_train = X[tr_indices]
    y_train = y[tr_indices]
    X_test = X[ts_indices]
    y_test = y[ts_indices]

    if not isinstance(X_train, np.ndarray):
        X_train.sort_indices()
        y_train.sort_indices()


    return X_train, y_train, X_test, y_test


def load_mlc_dataset(filename, header=True, concatbias=True):
    """
    This function extends 'load_svmlight_file' so that datasets that have a header
    are parsed correctly

    Args:
        @param filename: Path of the dataset.
        @param header: True if file has a header.
        @param concatbias: Whether to add bias.

    Returns:
        @returns X_train: csr_matrix(TODO check) which contains the features of each point.
        @returns y_train: csr_matrix(TODO check) which contains the labels of each point.
        @returns header_info: False if there is no header, contains an array which contains
                    0: Number of training examples
                    1: Feature Dimensionality
                    2: Label Dimensionality
    """
    logging.info("Started Loading Dataset")
    f = open(filename, mode='rb')

    header_info = False
    if header:
        header_info = f.readline().split()
        Xsparse, y = load_svmlight_file(
            f=f,
            multilabel=True,

        )
        if concatbias:
            one = sp.csr_matrix(np.ones(shape=(Xsparse.shape[0], 1)))
            X = scipy.sparse.hstack([one, Xsparse], format="csr")
        else:
            X = Xsparse

        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        ult = (sp.lil_matrix((DATASET_SIZE, LABEL_NUMBER)))
        # iterator = tqdm.trange(0, DATASET_SIZE)
        iterator = range(0, DATASET_SIZE)
        for i in iterator:
            ind_of_labels = np.asarray(y[i], dtype=int)
            ult[i, ind_of_labels] = 1
        y = ult.asformat('csr')

    else:
        X, y = load_svmlight_file(
            f=f,
            multilabel=True,
        )
    f.close()
    logging.info("Finished Loading")
    return X, y


@profile
def grad_check(X, W, y):
    """
    Checks the validation of gradient versus a numerical approximation
    @param X: Training examples
    @param W: Weight vector
    @param y: True Categories of the training examples X
    @return Maximum absolute error of calculated gradient
    """
    if scipy.sparse.issparse(X):
        calc_gradient = gradient_sp(X=X, W=W, y=y)
    else:
        # true_gradient = gradient(X=X, W=W, y=y)
        raise NotImplemented

    epsilon = 1e-6
    num_grad = np.zeros(W.shape[0])
    # iterr = tqdm.trange(0, W.shape[0])
    iterr = np.arange(0, W.shape[0])
    for k in iterr:
        W_tmpP = np.zeros(W.shape)
        W_tmpP[:] = W

        W_tmpM = np.zeros(W.shape)
        W_tmpM[:] = W

        W_tmpP[k] = W_tmpP[k] + epsilon
        if scipy.sparse.issparse(X):
            Ewplus = log_likelihood_sp(X, W_tmpP, y)
        else:
            Ewplus = log_likelihood(X, W_tmpP, y)

        W_tmpM[k] = W_tmpM[k] - epsilon
        if scipy.sparse.issparse(X):
            Ewminus = log_likelihood_sp(X, W_tmpM, y)
        else:
            Ewminus = log_likelihood(X, W_tmpM, y)

        num_grad[k] = np.divide((np.subtract(Ewplus, Ewminus)), np.multiply(2, epsilon))
    calc_gradient = np.reshape(calc_gradient, calc_gradient.shape[0])
    calc_gradient.reshape((W.shape[0], 1))
    num_grad.reshape((W.shape[0], 1))

    abs_max = np.linalg.norm(calc_gradient - num_grad) / np.linalg.norm(calc_gradient + num_grad)
    return abs_max


def plot_linreg_results(X, W, y, preds, lossHistory):
    # print("Score " + str(accuracy_score(y_true=y, y_pred=preds)))

    Y = (-W[0] - (W[1] * X)) / W[2]

    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
    plt.plot(X, Y, "r-")

    fig = plt.figure()

    plt.plot(np.arange(0, len(lossHistory)), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=ShuffleSplit(n_splits=20, test_size=0.1),
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Source : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    Notee
    --------------
    Dataset must be big enough

    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt


def plot_roc_curve(clf, dataset=sklearn.datasets.load_breast_cancer()):
    """
    Plot ROC curve.
    @param clf: Classifier
    @param dataset: Dataset
    @return:
    """
    iris = dataset
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(clf, dataset=datasets.load_breast_cancer(), X=None, y=None):
    """
    Plot confusion matrix.
    @param clf: Classifier
    @param dataset: Dataset
    @return:
    """
    if X is None:
        iris = dataset
        X = iris.data
        y = iris.target
        class_names = iris.target_names
    else:
        class_names = ['0', '1']

    X_train, X_test, y_train, y_test = _sklearn_split(X, y, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    def _plot_confusion_matrix(cm, classes,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # if normalize:
        #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #     #print("Normalized confusion matrix")

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                           title='Normalized confusion matrix')

    plt.show()


def report(results, n_top=3, duplicates=False):
    t = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            t += 1
            if t == n_top and not duplicates:
                break


def report_params(results, n_top=3, duplicates=False):
    t = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            return (results['params'][candidate])


def generate_load_cache(filename, X_train, y_train, batch_size):
    """
    Split X_train, y_train, in batches and save them.
    @param filename
    @param X_train:
    @param y_train:
    @param batch_size:
    @return:
    """
    try:
        cache = np.load(filename + "_batches.npz")['a']
    except IOError:
        # shuffle X_train, y_train
        data_size = y_train.shape[0]

        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y_train[shuffle_indices]
        shuffled_tx = X_train[shuffle_indices]

        X_train = shuffled_tx
        y_train = shuffled_y
        # generate batches -> insert to list
        cache = list(batch_iter(y_train, X_train, batch_size))
        # save list as filename
        np.savez(filename + "_batches.npz", a=cache)
    return cache


def shuffle_dataset(X, y, copy=False):
    """
    Shuffles X and y
    @param X:
    @param y:
    @param copy:
    @return:
    """
    data_size = y.shape[0]

    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = X[shuffle_indices]

    if copy:
        X_train = shuffled_tx.copy()
        y_train = shuffled_y.copy()
    else:
        X_train = shuffled_tx
        y_train = shuffled_y

    return X_train, y_train


def batch_iter(y, X, batch_size, shuffle=False):
    if shuffle:
        y, X = shuffle_dataset(y, X)

    for i in np.arange(0, X.shape[0], int(batch_size)):
        limit = (i + batch_size)
        if limit > X.shape[0]:
            limit = X.shape[0]
        if scipy.sparse.issparse(X):
            yield (X[i:limit, :], y[i:limit, :])
        else:
            yield (X[i:limit, :], y[i:limit, :])


def batch_iter_linreg_test(y, X, batch_size, shuffle=False):
    """
    Created for testing the linear regression classifier
    """
    for i in np.arange(0, X.shape[0], int(batch_size)):
        limit = (i + batch_size)
        if limit > X.shape[0]:
            limit = X.shape[0]
        if scipy.sparse.issparse(X):
            yield (X[i:limit, :], y[i:limit])
        else:
            yield (X[i:limit, :], y[i:limit])


def size(x, str1=' '):
    """
    Prints shape of x along with a message.
    @param x:
    @param str1: A message to print with the shape.
    """
    print(str(x.shape) + ' ' + str1)
    return 0


def tic():
    """
    MATLAB's tic.
    """
    return time()


def toc(start_time, str1=" ", print_res=True):
    """
    MATLAB's toc.
    @param start_time: Time supplied by tic
    @param str1: Extra string to be included in the beginning of print statement
    """
    if print_res:
        print(str1 + " " + "--- %s seconds ---" % (time() - start_time))
    return time() - start_time


def typu(x):
    """
    Print type of object.
    @param x:
    @return:
    """
    print(type(x))


def save_sparse_csr(filename, array):
    """
    Saves a sparse array in npz format in current directory.
    @param filename:
    @param array:
    @return:
    """
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    """
    Loads a sparse array from an npz file in current directory.
    @param filename:
    @return:
    """
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


# noinspection PyUnresolvedReferences
@profile
def concatenate_csr_matrices_by_columns(matrix1, matrix2):
    """
    Concatenates two csr sparse matrices in a more efficient way than hstack
    @param matrix1:
    @param matrix2:
    @return:
    """
    csr = isinstance(matrix1, scipy.sparse.csr_matrix)
    if isinstance(matrix2, np.ndarray):
        if csr:
            matrix2 = sp.csr_matrix(matrix2)
        else:
            matrix2 = sp.csc_matrix(matrix2)

    if csr:
        matrix1 = matrix1.T.asformat('csr')
        # noinspection PyUnresolvedReferences
        matrix2 = matrix2.T.asformat('csr')
    else:
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    if csr:
        return sp.csr_matrix(
            (new_data,
             new_indices,
             new_ind_ptr)
        ).T.asformat('csr')
    else:
        return sp.csr_matrix(
            (new_data,
             new_indices,
             new_ind_ptr)
        ).T.asformat('csc')


def load_delicious(feature=None):
    """
    This function loads delicious dataset with 16105 training examples and 500 features.
    @param feature: number of label to return if it is not set return all labels
    @return: X_train, y_train, X_test, y_test
    """
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

    if feature is not None:
        return X_train, y_train[:, feature], X_test, y_test[:, feature]

    return X_train, y_train, X_test, y_test


def load_breast_cancer(split=True, train_test_ratio=0.8):
    """
    This function returns the sklearn toy breast cancer dataset.
    @param split:
    @param train_test_ratio:
    @return:
    """
    (X, y) = sklearn.datasets.load_breast_cancer(True)
    X = sklearn.preprocessing.normalize(X)
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_test_ratio)
    if split:
        return X_train, y_train, X_test, y_test
    return X, y
