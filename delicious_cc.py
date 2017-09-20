import numpy as np
import scipy.sparse as sp
from MlcClassifierChains import MlcClassifierChains
from MlcLinReg import MlcLinReg
from helpers import load_mlc_dataset, tic, toc
from MlcScore import score_accuracy
from scipy.io import savemat
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit

# Comment when debuging with line profiler
profile = lambda f: f


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

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
    return plt


def delicious_mlc(learning_rate=0.0499,
                  iterations=300,
                  sparse=True,
                  verbose=False,
                  grad_check=False,
                  batch_size=2048,
                  alpha=0.5,
                  velocity=0.9):
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
                              learning_rate=learning_rate,
                              iterations=iterations,
                              sparse=True,
                              verbose=False,
                              batch_size=batch_size,
                              alpha=0.5,
                              velocity=0.9)

    # mlc.fit(X_train, y_train)

    title = "Learning Curve"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(mlc, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

    # from scipy.stats import bernoulli
    # y_pred = sp.csr_matrix(mlc.predict(X_test))
    # y_random = bernoulli.rvs(.5, size=y_pred.shape)
    #
    # print "########################"
    # print "random precision"
    # print(score_accuracy(y_random, y_test))
    # print "predicted precision"
    # print(score_accuracy(y_pred, y_test))
    # print "params"
    # print "learning rate : " + str(learning_rate)
    # print "batch size : " + str(batch_size)
    # print "iterations : " + str(iterations)
    # print "########################"

    # savemat('y_test.mat', {'y_test':y_test})
    # savemat('y_pred.mat', {'y_pred':y_pred})
    toc(ti)
    # fig = plt.figure()
    #
    # plt.plot(np.arange(0, len(mlc.lossHistory)), mlc.lossHistory)
    # fig.suptitle("Training Loss")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss")
    # plt.show()


# RandomGridSearch
# import scipy.stats
# for n in range(15):
#     l = np.random.uniform(0.001, 0.06)
#     batch = np.random.randint(512, 2000)
#     delicious_mlc(learning_rate=l,
#                   batch_size=batch,
#                   iterations=150
#                   )

delicious_mlc(learning_rate=0.0419799478595,
              iterations=100,
              batch_size=1023
              )
