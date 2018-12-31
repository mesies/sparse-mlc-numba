import logging
from abc import ABCMeta

import numpy as np
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six

import MlcScore
from MlcLinReg import MlcLinReg
from helpers import shuffle_dataset
from helpers.profile_support import profile


# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler


class MlcClassifierChains(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self,
                 learning_rate=0.0499,
                 iterations=1000,
                 sparse=True,
                 verbose=0,
                 batch_size=300,
                 parameter_pack=None,
                 limit_iterations_for_debug=None):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.sparse = sparse
        self.verbose = verbose
        self.batch_size = batch_size
        self.parameter_pack = parameter_pack

        self.trained = []
        self.classifier_type = MlcLinReg
        self.label_dim = -1
        self.lossHistory = []

        self.limit_iterations_for_debug = limit_iterations_for_debug
        logging.basicConfig(level=logging.WARNING)

    @profile
    def fit(self, X_train, y_train):
        """
        Train with X, y0 -> keep weights in self.weights[0] \n
        Train with X, y1 -> keep weights in self.weights[1] \n
        ...\n
        Train with X, yn -> keep weights in self.weights[n]\n
        @param X_train: Features of training examples.
        @param y_train: Labels of training examples
        """
        logging.info("***************************************************")
        logging.info("       Commencing Classifier Chain training")
        logging.info("***************************************************")

        shuffle_dataset(X_train, y_train)
        X = X_train


        # Add label 0 to features
        # X = concatenate_csr_matrices_by_columns(X_train, y_train[:, 0])

        self.label_dim = y_train.shape[1]

        _init = False
        # if self.verbose > 0:
        iterator = tqdm.tqdm(range(0, self.label_dim))
        # iterator = range(1, self.label_dim)

        for i in iterator:
            # Train Classifier i
            y = y_train[:, i]

            # Create and fit an instance of chosen classifier with chosen arguments and train it
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
            # X = concatenate_csr_matrices_by_columns(X, y)

            if self.limit_iterations_for_debug != None:
                if i == self.limit_iterations_for_debug: exit(0)

        return self

    def predict(self, X_test):
        """
        Predicts the labels of X_test
        @param X_test:
        @return:
        """
        logging.info("***************************************************")
        logging.info("       Commencing Classifier Chain predicting")
        logging.info("***************************************************")

        # Predict Label 0
        i = 0
        X = X_test

        # Retrieve trained classifier for label 0
        clf = self.trained[i]

        # Make prediction
        y = clf.predict(X)
        result = np.zeros((X_test.shape[0], self.label_dim))

        result[:, i] = y

        # Concatenate result to X
        # X = sp.hstack([X, sp.csr_matrix(y)], format="csr")

        # iterator = tqdm.trange(1, self.label_dim)
        iterator = range(1, self.label_dim)
        for i in iterator:
            # Predict Label i

            # Retrieve trained classifier for label i
            clf = self.trained[i]

            # Make prediction
            y = clf.predict(X)

            result[:, i] = y

            # Concatenate result to X
            # X = sp.hstack([X, sp.csr_matrix(y)], format="csr")

        return result

    def score(self, X, y, sample_weight=None):
        return MlcScore.score_accuracy(y_predicted=self.predict(X), y_real=y)

    def get_params(self, deep=True):
        return super(MlcClassifierChains, self).get_params(deep)
