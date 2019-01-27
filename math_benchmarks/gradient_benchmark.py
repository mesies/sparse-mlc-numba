import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sklearn

import helpers
import sparse_math_lib.gradient
import sparse_math_lib.logloss


def generate_data(rows, features, sparse):
    X = scipy.sparse.rand(rows, features, density=0.001, format='csr', dtype=float)
    y = scipy.sparse.rand(rows, features, density=0.001, format='csr', dtype=float)

    if sparse:
        return X, y

    (X, y) = sklearn.datasets.make_blobs(
        n_samples=rows,
        n_features=features,
        centers=2,
        cluster_std=2,
        random_state=777)

    return X, y


rows = np.array([10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5])
rows_static = 10 ** 6

features = np.array([10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5, 10 ** 5])
features_static = 10 ** 1

times_grad = []
times_logloss = []
times_logloss2 = []

sizes = []

i = 1
for rows_n in rows:
    X, y = generate_data(rows_n * i, features_static, True)
    sizes.append(rows_n * i)

    W = (np.random.uniform(size=(X.shape[1], 1)))

    t = helpers.tic()
    grad = sparse_math_lib.gradient.gradient_sp(X, W, y)
    times_grad.append(helpers.toc(t))

    t2 = helpers.tic()
    logloss = sparse_math_lib.logloss.log_likelihood_sp(X, W, y)
    times_logloss.append(helpers.toc(t2))

    t3 = helpers.tic()
    logloss2 = sparse_math_lib.logloss.log_likelihood_numba(X, W, y)
    times_logloss2.append(helpers.toc(t3))

    i += 1

plt.plot(sizes, times_grad, label="MLC-SGD Mini-Batch Gradient")
plt.plot(sizes, times_logloss, label="MLC-SGD Mini-Batch LogLoss")
plt.plot(sizes, times_logloss2, label="MLC-SGD Mini-Batch LogLoss Numba")
plt.xlabel('Rows Size')
plt.ylabel('Time')
plt.legend()
plt.show()
