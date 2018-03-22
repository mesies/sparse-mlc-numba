import logging
from abc import ABCMeta

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six
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


class MlcLinReg(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self,
                 learning_rate=0.2,
                 iterations=200,
                 verbose=False,
                 grad_check=False,
                 batch_size=512,
                 l_one=0.01):

        self.batch_size = batch_size
        self.verbose = verbose
        self.grad_check = grad_check
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}
        self.lossHistory = np.zeros(iterations)
        self.l_one = l_one
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)

    def fit(self, X, y):
        """
        Fits the classifier using X and y as training examples
        @param X:
        @param y:
        @return:
        """

        logging.info("Started Fitting Dataa")
        self.w = (np.random.uniform(size=(X.shape[1], 1)))

        if self.grad_check:
            logging.info("Commencing Gradient Check")
            abs_max = (helpers.grad_check(X, self.w, y))
            return abs_max

        y, X = shuffle(y, X)
        if not isinstance(y, sp.csr_matrix):
            y = np.reshape(y, (y.shape[0], 1))

        if X.shape[0] != y.shape[0]:
            y = np.reshape(y, (y.shape[1], y.shape[0]))

        _x = sp.csr_matrix(X)
        _y = sp.csr_matrix(y)

        self.w = self.stochastic_gradient_descent_sparse(_x, _y)

        return self

    @profile
    def stochastic_gradient_descent_sparse(self, X, y):

        tolerance = 1e-3
        epochs = self.iterations
        batch_size = self.batch_size

        logging.info("Commencing sparse-aware SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.learning_rate)
        epoch_loss = []
        # learning rate e, momentum parameter a,
        self.lossHistory = np.zeros(self.iterations)

        batches = list(self.batch_iter(y, X, batch_size))
        W = self.w
        l_one = self.l_one
        L = np.array(self.learning_rate)

        d = 1e-8
        r1 = 0.9
        r2 = 0.999
        old_loss_ep = np.inf
        t = 0.
        s = 0.
        r = 0.
        for epoch in range(0, epochs):

            old_loss = np.inf
            grads = []
            epoch_loss = []
            shuffle_indices = np.random.permutation(np.arange(len(batches)))
            # l_one = 0

            for batch_ind in shuffle_indices:
                (sampleX, sampley) = batches[batch_ind]
                sampleX, sampley = helpers.shuffle_dataset(sampleX, sampley)

                loss = sparse_math_lib.logloss.log_likelihood_sp(X=sampleX, y=sampley, W=W)
                loss += (l_one * np.sum(np.abs(W)))
                assert loss is not np.nan

                epoch_loss.append(loss)

                t += 1

                gradient = (sparse_math_lib.gradient.gradient_sp(X=sampleX, W=(W), y=sampley))
                gradient += (l_one * np.sign(W))
                assert gradient is not np.nan

                grads.append(gradient)

                s = r1 * s + (1. - r1) * gradient
                r = r2 * r + ((1. - r2) * gradient) * gradient

                s_hat = s / (1. - (r1 ** t))
                r_hat = r / (1. - (r2 ** t))

                velocity = -L * (s_hat / (np.sqrt(r_hat) + d))
                assert velocity is not np.nan
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss
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
            limit = 1e-3
            if (new_loss - old_loss_ep) >= limit and epoch > 20:
                break
            old_loss_ep = np.average(epoch_loss)

        return self.w

    def batch_iter(self, y, tx, batch_size, shuffle=False):
        """
         Note:
            See helpers.batch_iter
        """
        return helpers.batch_iter(y, tx, batch_size, shuffle)

    def predict(self, X):
        logging.info("Predicting Labels")
        y = mathutil.sigmoid((X.dot(self.w)))
        y = np.around(y)
        return np.reshape(y, (y.shape[0],))

    def predict_proba(self, X):
        c1_prob = mathutil.sigmoid((X.dot(self.w)))
        c2_prob = 1. - c1_prob
        return np.concatenate((c2_prob, c1_prob), axis=1)


    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return sklearn.metrics.f1_score(y, y_pred)

    def get_params(self, deep=True):
        return super(MlcLinReg, self).get_params(deep)

    def plot_log_loss(self):
        """
        Plot log loss of this model.
        """

        fig = plt.figure()
        plt.plot(np.arange(0, len(self.get_loss_history())), self.get_loss_history())
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()

    def get_loss_history(self):
        loss_h = []
        for i in range(0, len(self.lossHistory)):
            if self.lossHistory[i] == 0.0:
                break
            loss_h.append(self.lossHistory[i])
        return loss_h
