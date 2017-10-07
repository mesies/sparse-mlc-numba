import numpy as np
import scipy.sparse as sp
import logging
from helpers import concatenate_csr_matrices_by_columns
from MlcLinReg import MlcLinReg
import tqdm
import MlcScore
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six
from abc import ABCMeta, abstractmethod

# Comment when debugging with line profiler
profile = lambda f: f


class MlcClassifierChains(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self,
                 learning_rate=0.0499,
                 iterations=1000,
                 sparse=True,
                 verbose=False,
                 grad_check=False,
                 batch_size=300,
                 alpha=0.5,
                 velocity=1,
                 cache=None):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.sparse = sparse
        self.verbose = verbose
        self.grad_check = grad_check
        self.batch_size = batch_size
        self.alpha = alpha
        self.velocity = velocity
        self.cache = cache

        self.trained = []
        self.classifier_type = MlcLinReg
        self.label_dim = -1
        self.lossHistory = []
        logging.basicConfig(level=logging.WARNING)

    @profile
    def fit(self, X_train, y_train):
        """
        Train with X, y0 -> keep weights in self.weights[0]
        Train with X.concat(y0), y1 -> keep weights in self.weights[1]
        ...
        Train with X.concat(y0,...,yn-1), yn -> keep weights in self.weights[n]
        :param X_train: Features of training examples.
        :param y_train: Labels of training examples
        """
        logging.info("***************************************************")
        logging.info("       Commencing Classifier Chain training")
        logging.info("***************************************************")

        if self.cache is None:
            ##Shuffle Data
            data_size = y_train.shape[0]

            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y_train[shuffle_indices]
            shuffled_tx = X_train[shuffle_indices]

            X_train = shuffled_tx
            y_train = shuffled_y

        ## Train Classifier 0
        X = X_train
        y = y_train[:, 0]

        # Create an instance of chosen classifier with chosen arguments
        clf = self.classifier_type(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            iterations=self.iterations,
            cache=self.cache)
        clf.fit(X, y)

        # Save the trained instance
        self.trained.append(clf)

        # Add label 0 to features
        X = concatenate_csr_matrices_by_columns(X_train, y_train[:, 0])

        self.label_dim = y_train.shape[1]

        _init = False

        iterator = tqdm.trange(1, self.label_dim)
        # iterator = range(1, self.label_dim)
        for i in iterator:
            ## Train Classifier i
            y = y_train[:, i]

            # Create and fit an instance of chosen classifier with chosen arguments and trains it
            clf = self.classifier_type(learning_rate=self.learning_rate,
                                       batch_size=self.batch_size,
                                       iterations=self.iterations)
            clf.fit(X, y)

            if not _init:
                self.lossHistory = np.zeros(len(clf.lossHistory))
                _init = True
            self.lossHistory = self.lossHistory + np.asarray(clf.lossHistory) / 1000

            # Save the trained instance
            self.trained.append(clf)

            # Add label i to features
            X = concatenate_csr_matrices_by_columns(X, y)
            if i == 30: exit(0)
        return self

    @profile
    def predict(self, X_test):
        """
        Predicts the labels of X_test
        :param X_test:
        :return:
        """
        logging.info("***************************************************")
        logging.info("       Commencing Classifier Chain predicting")
        logging.info("***************************************************")

        ## Predict Label 0
        i = 0
        X = X_test

        # Retrieve trained classifier for label 0
        clf = self.trained[i]

        # Make prediction
        y = clf.predict(X)
        result = np.zeros((X_test.shape[0], self.label_dim))
        result[:, i] = y
        y = y.reshape((y.shape[0], 1))

        # Concatenate result to X
        X = sp.hstack([X, sp.csr_matrix(y)], format="csr")

        # iterator = tqdm.trange(1, self.label_dim)
        iterator = range(1, self.label_dim)
        for i in iterator:
            ## Predict Label i

            # Retrieve trained classifier for label i
            clf = self.trained[i]

            # Make prediction
            y = clf.predict(X)
            result[:, i] = y
            y = y.reshape((y.shape[0], 1))

            # Concatenate result to X
            X = sp.hstack([X, sp.csr_matrix(y)], format="csr")

        return result

    def score(self, X, y, sample_weight=None):
        return MlcScore.score_accuracy(ypredicted=self.predict(X), yreal=y)

    def get_params(self, deep=True):
        return super(MlcClassifierChains, self).get_params(deep)
