import numpy as np
import scipy.sparse as sp
import logging
import tqdm


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

        X = X_train
        y = y_train[:, 0]
        clf = self.classifier_type(**self.args)
        clf.fit(X, y)
        self.trained.append(clf)
        X = sp.hstack([X_train, y_train[:, 0]], format="csr")

        self.label_dim = y_train.shape[1]

        for i in tqdm.trange(1, self.label_dim):
            y = y_train[:, i]
            clf = self.classifier_type(**self.args)
            clf.fit(X, y)
            self.trained.append(clf)
            X = sp.hstack([X, y_train[:, i]], format="csr")
            if i==200: exit()

    def predict(self, X_test):
        logging.info("***************************************************")
        logging.info("       Commencing Classifier Chain predicting")
        logging.info("***************************************************")
        result = sp.lil_matrix((X_test.shape[0], self.label_dim))
        i = 0
        X = X_test
        y = self.trained.pop(i).predict(X)
        result[i, :] = y
        X = sp.hstack([X, result[i, :]], format="csr")

        for i in tqdm.trange(1, self.label_dim):
            y = self.trained.pop(i).predict(X)
            result[i, :] = y
            X = sp.hstack([X, result[i, :]], format="csr")
