import numpy as np
import scipy.sparse as sp
import logging
from helpers import concatenate_csr_matrices_by_columns
import tqdm

# Comment when debuging with line profiler
profile = lambda f: f


class MlcClassifierChains:
    def __init__(self,
                 classifier,
                 **args):
        self.args = args
        self.trained = []
        self.classifier_type = classifier
        self.label_dim = -1
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

        ## Train Classifier 0
        X = X_train
        y = y_train[:, 0]

        # Create an instance of chosen classifier with chosen arguments
        clf = self.classifier_type(**self.args)
        clf.fit(X, y)

        # Save the trained instance
        self.trained.append(clf)

        # Add label 0 to features
        X = concatenate_csr_matrices_by_columns(X_train, y_train[:, 0])

        self.label_dim = y_train.shape[1]

        for i in tqdm.trange(1, self.label_dim):
            ## Train Classifier i
            y = y_train[:, i]

            # Create and fit an instance of chosen classifier with chosen arguments and trains it
            clf = self.classifier_type(**self.args)
            clf.fit(X, y)

            # Save the trained instance
            self.trained.append(clf)

            # Add label i to features
            X = concatenate_csr_matrices_by_columns(X, y)
            # if i == 40: exit(1)

    @profile
    def predict(self, X_test):
        """
        Predicts te labes of X_test
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

        for i in tqdm.trange(1, self.label_dim):
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
