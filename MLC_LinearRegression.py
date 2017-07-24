import math
import logging
import numpy as np
import helpers

"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MLC_LinearRegression:
    def __init__(self, learning_rate=0.001, iterations=1000, debug=True):

        self.l = learning_rate
        self.iterations = iterations
        self.w = {}
        self.debug = debug
        self.lossHistory = []
        if debug:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.INFO)

    def fit(self, X, y):
        logging.info("Started Fitting Dataa")
        self.w = np.random.uniform(size=(X.shape[1],))
        logging.debug("Commencing Gradient Check")
        #logging.debug(helpers.grad_check(X, self.w, y))

        self.w = self.gradient_decent(X, y, epochs=self.iterations)
        return self.w

    def gradient_decent(self, X, y, tolerance=1e-3, epochs=10000):
        old_loss = np.inf
        for epoch in np.arange(0, epochs):

            logging.info("INFO: Commencing next epoch %i", epoch)

            gradient = helpers.gradient(X, self.w, y)
            loss = helpers.log_likelihood(X, self.w, y)
            self.lossHistory.append(loss)
            logging.info("INFO: epoch #{}, loss={} , gradient={}".format(epoch + 1, loss, gradient))

            if np.abs(loss - old_loss) < tolerance:
                break
            old_loss = loss

            self.w = self.w - self.l * gradient

        return self.w

    def predict(self, X):
        logging.info("Predicting Labels")
        #X_train = np.c_[np.ones((X.shape[0])), X]
        y = helpers.sigmoid(np.dot(X, self.w))
        y = np.around(y)
        return y

