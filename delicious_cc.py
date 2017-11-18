import os

import sklearn.model_selection
from scipy.io import savemat
from scipy.stats import randint
from sklearn.model_selection import ShuffleSplit

from MlcClassifierChains import MlcClassifierChains
from MlcScore import score_accuracy
from helpers import *

# Comment when debuging with line profiler
profile = lambda none: none


def load_delicious():
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

    return X_train, y_train, X_test, y_test


ti = tic()

X_train, y_train, X_test, y_test = load_delicious()

X_train_s, y_train_s = shuffle_dataset(X_train, y_train)

# Create train test split
X_train, y_train, X_test, y_test = split_train_test(X_train_s, y_train_s)

# learning_rate = 0.0227
# iterations = 100
# batch_size = 8000
n = 1
for i in range(0, n):
    learning_rate = np.random.uniform(0.005, 0.02)
    iterations = np.random.randint(300, 500)
    batch_size = np.random.randint(1000, 5000)

    mlc = MlcClassifierChains(learning_rate=learning_rate,
                              iterations=iterations,
                              sparse=True,
                              verbose=True,
                              batch_size=batch_size,
                              alpha=0.5,
                              velocity=0.9)

    mlc.fit(X_train, y_train)
    y_pred = mlc.predict(X_test)
    print(score_accuracy(y_pred, y_test))
    with open("results_random.txt", "a") as result_file:
        result_file.write(
            "% " +
            str(score_accuracy(y_pred, y_test)) + "  " +
            str(learning_rate) + " " + str(iterations) + "  " +
            str(batch_size) +
            " %")

exit(1)

savemat('y_test.mat', {'y_test': y_test})
savemat('y_pred.mat', {'y_pred': y_pred})

title = "Learning Curve"
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
if __name__ == '__main__':
    plot = plot_learning_curve(MlcClassifierChains(learning_rate=learning_rate,
                                                   iterations=iterations,
                                                   sparse=True,
                                                   verbose=False,
                                                   batch_size=batch_size,
                                                   alpha=0.5,
                                                   velocity=0.9), title, X_train, y_train, ylim=(0.7, 1.01),
                               cv=cv, n_jobs=1)
    plot.savefig('learning_curve' + str(learning_rate) + '_' + str(batch_size) + '.png')

fig = plt.figure()

plt.plot(np.arange(0, len(mlc.lossHistory)), mlc.lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plot.savefig('loss_history' + str(learning_rate) + '_' + str(batch_size) + '.png')


def random_grid_search(X_train, y_train):
    if __name__ == '__main__':
        gs = sklearn.model_selection.RandomizedSearchCV(estimator=mlc,
                                                        param_distributions={
                                                            'learning_rate': np.arange(0.004, 0.05, 0.001).tolist(),
                                                            "iterations": randint(250, 400),
                                                            "batch_size": randint(1000, 6000)
                                                        },
                                                        n_iter=5,
                                                        cv=3,
                                                        n_jobs=1,
                                                        verbose=20)
        gs.fit(X_train, y_train)
        report(gs.cv_results_)
