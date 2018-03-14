import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
import sklearn
from scipy.stats import randint as sp_randint
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit, KFold, RandomizedSearchCV

import helpers
from MlcLinReg import MlcLinReg
from helpers import shuffle_dataset, split_train_test, tic, toc, load_delicious, plot_learning_curve

stdout = sys.stdout
# sys.stdout = open('delicious_random_grid_results.txt', 'w')

scores = list()
scores_sgd = list()
times = list()
times_sgd = list()
feature = 6
batch_size = 1024
iterations = 200

X_train, y_train, X_test, y_test = helpers.load_delicious(feature)
plot_learning_curve(
    estimator=MlcLinReg(learning_rate=0.005, iterations=100, batch_size=4),
    title="Learning Curve",
    X=(X_train).toarray(),
    y=(y_train).toarray(),
)

mlc1 = MlcLinReg(learning_rate=0.01, iterations=400, batch_size=2)
mlc1.fit(X_train, y_train)
y_pred = mlc1.predict(X_test)
mlc1.plot_log_loss()
print f1_score(y_pred=y_pred, y_true=y_test.toarray())
