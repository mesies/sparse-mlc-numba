import logging

import numpy as np
from sklearn.utils import shuffle

import helpers
import sparse_math_lib.gradient
import sparse_math_lib.logloss
from sparse_math_lib import mathutil

profile = lambda f: f
# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator so in order to edit
# as few lines as possible each time line_profiler is run a lambda is used
# Comment when debugging with line profiler
"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MlcLinReg:
    def __init__(self,
                 learning_rate=0.001,
                 iterations=100,
                 sparse=True,
                 verbose=False,
                 grad_check=False,
                 batch_size=256,
                 alpha=0.9,
                 velocity=1,
                 regularization=0.01):

        self.batch_size = batch_size
        self.verbose = verbose
        self.grad_check = grad_check
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}
        self.sparse = sparse
        self.lossHistory = np.zeros(iterations)
        self.alpha = alpha
        self.velocity = velocity
        self.l_one = regularization
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)

    def fit(self, X, y):
        """
        Fits the classifier using X and y as training examples
        :param X:
        :param y:
        :return:
        """

        logging.info("Started Fitting Dataa")
        self.w = (np.random.uniform(size=(X.shape[1], 1)))

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
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.learning_rate)
        epoch_loss = []
        # learning rate e, momentum parameter a,

        y, X = shuffle(y, X)

        batches = list(self.batch_iter(y, X, batch_size, True))
        W = self.w
        l_one = self.l_one
        L = np.array(self.learning_rate)
        velocity = np.array(self.velocity)
        alpha = np.array(self.alpha)

        old_loss_ep = np.inf

        for epoch in range(0, epochs):

            old_loss = np.inf
            grads = []
            epoch_loss = []
            shuffle_indices = np.random.permutation(np.arange(len(batches)))

            for batch_ind in shuffle_indices:
                (sampleX, sampley) = batches[batch_ind]
                # sampleX, sampley = helpers.shuffle_dataset(sampleX, sampley)

                loss = sparse_math_lib.logloss.log_likelihood_sp(X=sampleX, y=sampley, W=W) + (
                            l_one * np.sum(np.abs(W)))
                epoch_loss.append(loss)
                av = alpha * velocity
                gradient = sparse_math_lib.gradient.gradient_sp(X=sampleX, W=(W + (av)), y=sampley) + (
                            l_one * np.sign(W))
                grads.append(gradient)

                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                # Nesterov momentum
                velocity = (av) - (L * gradient)
                assert velocity.shape == self.w.shape

                W = W + velocity

            assert self.w.shape == W.shape
            self.w = W
            new_loss = np.average(epoch_loss)
            self.lossHistory[epoch] = new_loss
            # print("Ending epoch {}, average loss -> {} Epoch gradient AVG -> {}".format(
            #       epoch,
            #       np.average(epoch_loss),
            #       np.average(grads)))

            # Maybe needs tweaking
            limit = (((float(epochs) - (epoch ** 3)) ** 3) / (epochs))
            if limit < 0:
                limit = 1e-2
            if (new_loss - old_loss_ep) > limit:
                break
            old_loss_ep = np.average(epoch_loss)

        return self.w

    # In helpers remove dependency
    def batch_iter(self, y, tx, batch_size, shuffle=False):
        return helpers.batch_iter(y, tx, batch_size, shuffle)

    def predict(self, X):
        logging.info("Predicting Labels")
        y = mathutil.sigmoid((X.dot(self.w)))
        y = np.around(y)
        return y

    # dont use
    def stochastic_gradient_descent(self, X, y, tolerance, epochs=2000, batch_size=10):

        logging.info("Commencing SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.learning_rate)
        epoch_loss = []
        velocity = 0.9
        alpha = 0.5
        learning_rate = self.learning_rate
        for epoch in np.arange(0, epochs):
            old_loss = np.inf
            # Shuffle X, y
            indexes = np.arange(np.shape(X)[0])
            np.random.shuffle(indexes)
            X = X[indexes, :]
            y = y[indexes]
            for (sampleX, sampley) in helpers.batch_iter_linreg_test(y, X, batch_size):

                loss = sparse_math_lib.logloss.log_likelihood(X=sampleX, y=sampley.T, W=self.w)

                epoch_loss.append(loss)
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                gradient = sparse_math_lib.gradient.gradient(sampleX, self.w, sampley)

                # self.w = self.w - self.l * gradient

                velocity = (alpha * velocity) - (learning_rate * gradient)
                self.w = self.w + velocity

            self.lossHistory.append(np.average(epoch_loss))
            logging.info("Ending epoch %i, average loss -> %f", epoch, np.average(epoch_loss))

            # if np.abs(np.average(epochloss) - old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w
