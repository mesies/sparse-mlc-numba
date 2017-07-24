import logging

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from autograd.util import quick_grad_check

import helpers

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
(X, y) = sklearn.datasets.make_blobs(
    n_samples=250,
    n_features=2,
    centers=2,
    cluster_std=1.05,
    random_state=20)

logging.basicConfig(filename='test.log', filemode='w', level=logging.INFO)

X = np.c_[np.ones((X.shape[0])), X]

print("[INFO] starting training...")

W = np.random.uniform(size=(X.shape[1],))

# initialize a list to store the loss value for each epoch
lossHistory = []
epochs = 1000
l = 0.001
sigm = np.vectorize(helpers.sigmoid)

logging.info("#####################  X ")
logging.info(X)

helpers.grad_check(X, W, y)
weights = np.array([0.0, 0.0, 0.0])
#quick_grad_check(fun=helpers.log_likelihood, arg0=X, extra_args=(weights, y))

old_loss = np.inf
time = helpers.tic()

for epoch in np.arange(0, epochs):
    logging.info("Commencing next epoch %i", epoch)

    xw = X.dot(W)
    sigm_wt_x = sigm(xw)

    gradient = helpers.gradient(X, W, y)
    loss = helpers.log_likelihood(X, W, y)

    if np.abs(loss - old_loss) < (10 ** -4):
        break

    old_loss = loss

    logging.info("#####################  W epoch %i", epoch)
    logging.info(W.shape)
    logging.info(W)
    logging.info("#####################  XW epoch %i", epoch)
    logging.info(xw.shape)
    logging.info(xw)
    logging.info("#####################  sigmXW epoch %i", epoch)
    logging.info(sigm_wt_x)

    lossHistory.append(loss)

    logging.info("#####################  grad epoch %i", epoch)
    logging.info(gradient)
    print("[INFO] epoch #{}, loss={:.7f} , gradient={}".format(epoch + 1, loss, gradient))

    Wold = W
    W = W - l * gradient

    logging.info("#####################  W after Up epoch %i", epoch)
    logging.info(W)
    logging.info(Wold)

helpers.toc(time)

for i in np.random.choice(250, 10):
    activation = sigm(X[i].dot(W))

    label = 0 if activation < 0.5 else 1

    print("activation={}; predicted_label={}, true_label={}".format(
        activation, label, y[i]))

Y = (-W[0] - (W[1] * X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")
# construct a figure that plots the loss over time
fig = plt.figure()
if epoch == 999:
    epoch = 1000
plt.plot(np.arange(0, epoch), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
