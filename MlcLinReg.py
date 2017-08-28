import logging
import numpy as np
import helpers
import mathutil

# Comment when debuging with line profiler
profile = lambda f: f


"""
This is an implementation of Linear Regression with SGD solver aiming at performance when training
examples matrix is a sparse matrix
"""


class MlcLinReg:
    def __init__(self,
                 learning_rate=0.0001,
                 iterations=1000,
                 sparse=False,
                 verbose=False,
                 grad_check=False,
                 batch_size=20,
                 alpha=0.5,
                 velocity=1):

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
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=__name__ + '.log', filemode='w', level=logging.DEBUG)

    def fit(self, X, y):
        logging.info("Started Fitting Dataa")
        self.w = np.random.uniform(size=(X.shape[1],))
        if self.grad_check:
            logging.info("Commencing Gradient Check")
            logging.info(helpers.grad_check(X, self.w, y))
            exit(0)
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

            gradient = mathutil.gradient(X, self.w, y)
            loss = mathutil.log_likelihood(X, self.w, y)
            self.lossHistory.append(loss)
            logging.info("epoch #{}, loss={} , gradient={}".format(epoch + 1, loss, gradient))

            if np.abs(loss - old_loss) < tolerance:
                break
            old_loss = loss

            self.w = self.w - self.l * gradient

        return self.w

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
            for (sampleX, sampley) in self.next_batch(X, y, batch_size):

                loss = mathutil.log_likelihood(X=sampleX, y=sampley.T, W=self.w)

                epoch_loss.append(loss)
                if np.abs(loss - old_loss) < tolerance:
                    break
                old_loss = loss

                gradient = mathutil.gradient(sampleX, self.w, sampley)

                self.w = self.w - self.l * gradient
            self.lossHistory.append(np.average(epoch_loss))
            logging.info("Ending epoch %i, average loss -> %f", epoch, np.average(epoch_loss))

            # if np.abs(np.average(epochloss) - old_loss_ep):
            #     break
            # old_loss_ep = np.average(epochloss)

        return self.w

    @profile
    def stochastic_gradient_descent_sparse(self, X, y, tolerance, epochs=2000, batch_size=10):
        logging.info("Commencing sparse-aware SGD")
        logging.info("Options : tol = %f, epochs = %f, learning rate = %f", tolerance, epochs, self.l)
        epoch_loss = []
        # learning rate e, momentum parameter a,

        for epoch in range(0, epochs):
            old_loss = np.inf

            grads = []
            for (sampleX, sampley) in self.batch_iter(X, y, batch_size):

                loss = mathutil.log_likelihood_sp(X=sampleX, y=sampley, W=self.w)
                gradient = mathutil.gradient_sp(sampleX, self.w, sampley)

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

    @staticmethod
    def batch_iter(y, tx, batch_size, num_batches=1, shuffle=False):
        data_size = y.shape[0]

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    def predict(self, X):
        logging.info("Predicting Labels")
        y = mathutil.sigmoid((X.dot(self.w)))
        y = np.around(y)
        return y
