import logging
import numpy as np
import helpers
import scipy.sparse

"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MLC_LinearRegression:
    def __init__(self, learning_rate=0.0001, iterations=1000, sparse=False, verbose=False, batch_size=20):
        self.batch_size = batch_size
        self.verbose = verbose
        self.l = learning_rate
        self.iterations = iterations
        self.w = {}
        self.sparse = sparse
        self.lossHistory = []
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)

    def fit(self, X, y):
        logging.info("Started Fitting Dataa")
        self.w = np.random.uniform(size=(X.shape[1],))
        if self.verbose:
            logging.info("Commencing Gradient Check")
            logging.info(helpers.grad_check(X, self.w, y))

        if self.sparse:
            self.w = self.stochastic_gradient_descent_sparse(X,
                                                             y,
                                                             epochs=self.iterations,
                                                             tolerance=1e-3,
                                                             batch_size=self.batch_size)
        else:
            self.w = self.stochastic_gradient_descent(X,
                                                      y,
                                                      epochs=self.iterations,
                                                      tolerance=1e-3,
                                                      batch_size=self.batch_size)
        return self.w

    def gradient_decent(self, X, y, tolerance=1e-3, epochs=10000):
        old_loss = np.inf
        for epoch in np.arange(0, epochs):

            logging.info("Commencing next epoch %i", epoch)

            gradient = helpers.gradient(X, self.w, y)
            loss = helpers.log_likelihood(X, self.w, y)
            self.lossHistory.append(loss)
            logging.info("epoch #{}, loss={} , gradient={}".format(epoch + 1, loss, gradient))

            if np.abs(loss - old_loss) < tolerance:
                break
            old_loss = loss

            self.w = self.w - self.l * gradient

        return self.w

    def stochastic_gradient_descent(self, X, y, tolerance, epochs=2000, batch_size=10):
        old_loss = np.inf
        old_loss_ep = np.inf
        logging.info("Commencing SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.l)
        epochloss = []
        for epoch in np.arange(0, epochs):
            old_loss = np.inf
            for (sampleX, sampley) in self.next_batch(X, y, batch_size):

                loss = helpers.log_likelihood(X=sampleX, y=sampley.T, W=self.w)

                epochloss.append(loss)
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                gradient = helpers.gradient(sampleX, self.w, sampley)

                self.w = self.w - self.l * gradient
            self.lossHistory.append(np.average(epochloss))
            logging.info("Ening epoch %i, average loss -> %f", epoch, np.average(epochloss))

            # if np.abs(np.average(epochloss) - old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w

    def stochastic_gradient_descent_sparse(self, X, y, tolerance, epochs=2000, batch_size=10):
        old_loss = np.inf
        old_loss_ep = np.inf

        logging.info("Commencing sparse-aware SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.l)
        epochloss = []
        for epoch in np.arange(0, epochs):
            old_loss = np.inf
            for (sampleX, sampley) in self.next_batch(X, y, batch_size):

                loss = helpers.log_likelihood(X=sampleX, y=sampley.T, W=self.w)

                epochloss.append(loss)
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                gradient = helpers.gradient(sampleX, self.w, sampley)

                self.w = self.w - self.l * gradient
            self.lossHistory.append(np.average(epochloss))
            logging.info("Ening epoch %i, average loss -> %f", epoch, np.average(epochloss))

            # if np.abs(np.average(epochloss) - old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w

    def next_batch(self, X, y, batchSize=5):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])

    def predict(self, X):
        logging.info("Predicting Labels")
        # X_train = np.c_[np.ones((X.shape[0])), X]
        y = helpers.sigmoid((X.dot(self.w)))
        y = np.around(y)
        return y
