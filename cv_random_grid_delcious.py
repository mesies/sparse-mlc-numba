import warnings
from time import time

import numpy as np
import scipy.stats as st
import tqdm
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

import helpers
from MlcLinReg import MlcLinReg

"""
This script runs a randomised grid search on all features of delicious dataset
"""
warnings.filterwarnings("ignore")

param_dist = {"learning_rate": st.uniform(0.001, 0.4),
              "iterations": sp_randint(50, 1000),
              "batch_size": sp_randint(2, 2000),
              "l_one": st.uniform(0.01, 0.5)
              }
best_params = np.zeros((501, 4))

# run randomized search
for feature in tqdm.tqdm(range(0, 501)):
    X_train, y_train, X_test, y_test = helpers.load_delicious(feature)
    clf = MlcLinReg()
    n_iter_search = 60
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(X_train.toarray(), y_train.toarray())
    conf = helpers.report_params(random_search.cv_results_, n_top=1)
    best_params[feature, :] = conf.values()

# 'delicious_best_params' has the following columns :
# learning_rate l_one iterations batch_size
np.savetxt("delicious_best_params.txt", best_params)

#
# warnings.filterwarnings("ignore")
#
# param_dist = {"learning_rate": st.uniform(0.001, 0.4),
#               "iterations": sp_randint(50, 1000),
#               "batch_size": sp_randint(2, 2000),
#               "l_one": st.uniform(0.01, 0.5)
#               }
# best_params = np.zeros((501, 4))
#
# # run randomized search
# X_train, y_train, X_test, y_test = helpers.load_breast_cancer()
# clf = MlcLinReg()
# n_iter_search = 60
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search)
#
# start = time()
# random_search.fit(X_train, y_train)
# helpers.report(random_search.cv_results_, n_top=3)
