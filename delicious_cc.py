import numpy as np
import scipy.sparse as sp
from MlcClassifierChains import MlcClassifierChains
from MlcLinReg import MlcLinReg
from helpers import load_mlc_dataset, tic, toc, plot_learning_curve, report
from MlcScore import score_accuracy
from scipy.io import savemat
from sklearn.model_selection import learning_curve, ShuffleSplit
import matplotlib.pyplot as plt

# Comment when debuging with line profiler
profile = lambda f: f


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

    mlc = MlcClassifierChains(learning_rate=learning_rate,
                              iterations=iterations,
                              sparse=True,
                              verbose=False,
                              batch_size=batch_size,
                              alpha=0.5,
                              velocity=0.9)

    import sklearn.model_selection
    from scipy.stats import uniform, randint as sp_randint
    if __name__ == '__main__':
        gs = sklearn.model_selection.RandomizedSearchCV(estimator=mlc,
                                                        param_distributions={
                                                            'learning_rate': np.arange(0.0, 10.0 + 0.0, 0.1).tolist(),
                                                            "iterations": sp_randint(100, 400),
                                                            "batch_size": sp_randint(256, 4096)
                                                            },
                                                        n_iter=15,
                                                        n_jobs=7)
        gs.fit(X_train, y_train)
        report(gs.cv_results_)
    exit(1)
    # mlc.fit(X_train, y_train)

    title = "Learning Curve"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    if __name__ == '__main__':
        plot = plot_learning_curve(mlc, title, X_train, y_train, ylim=(0.7, 1.01),
                                   cv=cv, n_jobs=7)
        plot.savefig('learning_curve' + str(learning_rate) + '_' + str(batch_size) + '.png')

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
    fig = plt.figure()

    plt.plot(np.arange(0, len(mlc.lossHistory)), mlc.lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plot.savefig('loss_history' + str(learning_rate) + '_' + str(batch_size) + '.png')


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
              batch_size=511
              )
