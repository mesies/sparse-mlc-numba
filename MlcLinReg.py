from abc import ABCMeta

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six

import helpers
import sparse_math_lib.gradient
import sparse_math_lib.logloss
from helpers.profile_support import profile
from sparse_math_lib import mathutil

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
                 grad_check=False,
                 batch_size=512,
                 l_one=0.01,
                 optimiser=None):

        self.batch_size = batch_size
        self.grad_check = grad_check
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}
        self.lossHistory = np.zeros(iterations)
        self.l_one = l_one
        if optimiser == "adam":
            self.d = 1e-8
            self.r1 = 0.9
            self.r2 = 0.999

            self.t = 0
            self.s = 0.
            self.r = 0.

            self.L = np.array(self.learning_rate)
            self.updateFunc = self.adam

        if optimiser == None:
            self.updateFunc = self.default


    def adam(self, W, gradient):
        self.t += 1
        self.s = self.r1 * self.s + (1. - self.r1) * gradient
        self.r = self.r2 * self.r + ((1. - self.r2) * gradient) * gradient

        s_hat = self.s / (1. - (self.r1 ** self.t))
        r_hat = self.r / (1. - (self.r2 ** self.t))

        velocity = -self.L * (s_hat / (np.sqrt(r_hat) + self.d))
        assert velocity is not np.nan
        assert velocity.shape == W.shape

        return W + velocity

    def momementum(self, W, gradient):
        return None

    def default(self, W, gradient):
        return W - self.learning_rate * gradient

    @profile
    def fit(self, X, y):
        """
        Fits the classifier using X and y as training examples
        @param X:
        @param y:
        @return:
        """

        self.w = (np.random.uniform(size=(X.shape[1], 1)))

        if self.grad_check:
            abs_max = (helpers.grad_check(X, self.w, y))
            return abs_max

        # y, X = shuffle(y, X)
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
        """
        ADAM Optimiser Implementation with custom stopping criterion
        @param X:
        @param y:
        @return:
        """
        tolerance = 1e-3
        epochs = self.iterations
        batch_size = self.batch_size

        epoch_loss = []

        # Generate CSR batches
        batches = list(self.batch_iter(y, X, batch_size))

        # Generate COO batches
        # batches_coo = list()
        # for batch in batches:
        #     batches_coo.append((batch[0].tocoo(), batch[1].tocoo()))

        # self.lossHistory = np.zeros(self.iterations * len(batches))
        self.lossHistory = np.zeros(self.iterations)

        l_one = self.l_one


        grads = []


        for epoch in range(0, epochs):

            epoch_loss = []
            shuffle_indices = np.random.permutation(np.arange(len(batches)))
            # l_one = 0

            for batch_ind in shuffle_indices:
                (sampleX, sampley) = batches[batch_ind]
                #sampleX, sampley = helpers.shuffle_dataset(sampleX, sampley)

                loss = sparse_math_lib.logloss.log_likelihood_sp(X=sampleX, y=sampley, W=self.w)
                # loss += (l_one * W.T.dot(W))
                assert loss is not np.nan

                epoch_loss.append(loss)

                gradient = (sparse_math_lib.gradient.gradient_sp(X=sampleX, W=self.w, y=sampley))
                #gradient += l_one * self.w
                assert gradient is not np.nan

                grads.append(gradient)

                self.w = self.updateFunc(self.w, gradient)

                # print("Ending epoch {},  loss -> {} velocity -> {}".format(
                #     t,
                #     loss,
                #     np.average(velocity)))

            new_loss = np.average(epoch_loss)
            self.lossHistory[epoch] = new_loss

            # Maybe needs tweaking
            old_loss_ep = np.average(epoch_loss)
            if self.stopping_criterion(new_loss, old_loss_ep, epoch): break

        return self.w

    def stopping_criterion(self, loss, old_loss, epoch):

        limit = 1e-3
        improvement_limit_percent = 0.0001
        epoch_limit = 150

        rules_and = []

        improvement_limit_percent_rule = ((loss - old_loss) / old_loss) < improvement_limit_percent or \
                                         abs(loss - old_loss) <= limit
        epoch_rule = epoch > epoch_limit

        positive_improvemnt_rule = abs(loss - old_loss) <= limit

        rules_and.append(positive_improvemnt_rule)
        rules_and.append(epoch_rule)

        stop = True
        for rule in rules_and:
            stop = rule and stop

        if (stop):
            print "Stopping MLC-SGD at " + str(loss) + " " + str(old_loss) + " epoch " + str(epoch)
            return True
        return False

    def batch_iter(self, y, tx, batch_size, shuffle=False):
        """
         Note:
            See helpers.batch_iter
        """
        return helpers.batch_iter(y, tx, batch_size, shuffle)

    def predict(self, X):

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
        plt.plot(np.arange(0, len(self.get_loss_history())), self.get_loss_history(), label="Log Loss")
        fig.suptitle("MLC-SGD Training Loss")
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

    def bayesianCVparams(self, X, y):
        import skopt
        import pandas as pd
        search_spaces = {
            "iterations": skopt.space.Integer(20, 400),
            "batch_size": skopt.space.Integer(20, 3000),
            "learning_rate": skopt.space.Real(0.001, 0.5, prior="log-uniform")}

        searchcv = skopt.BayesSearchCV(n_iter=400,
                                       estimator=self,
                                       search_spaces=search_spaces,
                                       n_jobs=5,
                                       cv=5)

        def status_print(optim_result):
            """Status callback durring bayesian hyperparameter search"""
            # Get all the models tested so far in DataFrame format
            all_models = pd.DataFrame(searchcv.cv_results_)
            # Get current parameters and the best parameters
            best_params = pd.Series(searchcv.best_params_)
            print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(len(all_models),
                                                                          np.round(searchcv.best_score_, 4),
                                                                          searchcv.best_params_))
            # Save all model results
            clf_name = searchcv.estimator.__class__.__name__
            all_models.to_csv(clf_name + "_cv_results.csv")

        searchcv.fit(X, y, callback=status_print)
