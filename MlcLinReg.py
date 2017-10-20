import logging

import numpy as np

import helpers
import sparse_math_lib.gradient
import sparse_math_lib.logloss
from sparse_math_lib import mathutil

# Comment when debugging with line profiler
profile = lambda f: f


"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MlcLinReg:
    def __init__(self,
                 learning_rate=0.0001,
                 iterations=1000,
                 sparse=True,
                 verbose=False,
                 grad_check=False,
                 batch_size=20,
                 alpha=0.5,
                 velocity=1,
                 cache=None):

        self.batch_size = batch_size
        self.verbose = verbose
        self.grad_check = grad_check
        self.l = learning_rate
        self.iterations = iterations
        self.w = {}
        self.sparse = sparse
        self.lossHistory = []
        self.alpha = alpha
        self.velocity = velocity
        self.cache = cache
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)

    @profile
    def fit(self, X, y):
        """
        Fits the classifier using X and y as training examples
        :param X:
        :param y:
        :return:
        """

        logging.info("Started Fitting Dataa")
        self.w = (np.random.uniform(size=(X.shape[1],)))

        if self.grad_check:
            logging.info("Commencing Gradient Check")
            abs_max = (helpers.grad_check(X, self.w, y))
            return abs_max

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
        return self

    @profile
    def stochastic_gradient_descent_sparse(self, X, y, tolerance, epochs=2000, batch_size=10):
        logging.info("Commencing sparse-aware SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.l)
        epoch_loss = []
        # learning rate e, momentum parameter a,

        # if self.cache is None:
        #     gen = self.batch_iter(y, X, batch_size)
        # else:
        #     gen = self.cache

        batches = list(self.batch_iter(y, X, batch_size))

        for epoch in range(0, epochs):
            old_loss = np.inf

            grads = []
            shuffle_indices = np.random.permutation(np.arange(len(batches)))

            for batch_ind in shuffle_indices:
                (sampleX, sampley) = batches[batch_ind]

                loss = sparse_math_lib.logloss.log_likelihood_sp(X=sampleX, y=sampley, W=self.w)
                gradient = sparse_math_lib.gradient.gradient_sp(sampleX, self.w, sampley)

                epoch_loss.append(loss)
                grads.append(gradient)

                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                self.velocity = (self.alpha * self.velocity) - (self.l * gradient)
                self.w = self.w + self.velocity

            self.lossHistory.append(np.average(epoch_loss))
            logging.info("Ending epoch %i, average loss -> %f Epoch gradient AVG -> %f", epoch, np.average(epoch_loss),
                         np.average(grads))
            # if(np.average(epoch_loss) > old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w

    @profile
    def batch_iter(self, y, tx, batch_size, num_batches=1, shuffle=False):
        return helpers.batch_iter(y, tx, batch_size)

    def predict(self, X):
        logging.info("Predicting Labels")
        y = mathutil.sigmoid((X.dot(self.w)))
        y = np.around(y)
        return y

    # dont use
    def stochastic_gradient_descent(self, X, y, tolerance, epochs=2000, batch_size=10):

        logging.info("Commencing SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.l)
        epoch_loss = []
        for epoch in np.arange(0, epochs):
            old_loss = np.inf
            # Shuffle X, y
            indexes = np.arange(np.shape(X)[0])
            np.random.shuffle(indexes)
            X = X[indexes, :]
            y = y[indexes]
            for (sampleX, sampley) in self.batch_iter(X, y, batch_size):

                loss = sparse_math_lib.logloss.log_likelihood(X=sampleX, y=sampley.T, W=self.w)

                epoch_loss.append(loss)
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                gradient = sparse_math_lib.gradient.gradient(sampleX, self.w, sampley)

                self.w = self.w - self.l * gradient
            self.lossHistory.append(np.average(epoch_loss))
            logging.info("Ending epoch %i, average loss -> %f", epoch, np.average(epoch_loss))

            # if np.abs(np.average(epochloss) - old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w
