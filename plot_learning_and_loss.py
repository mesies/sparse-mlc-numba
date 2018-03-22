import numpy as np
from sklearn.model_selection import train_test_split

import helpers
from MlcLinReg import MlcLinReg
from helpers import plot_learning_curve

scores = list()
scores_sgd = list()
times = list()
times_sgd = list()
feature = 2
batch_size = 1024
iterations = 200

X_train, y_train, X_test, y_test = helpers.load_delicious(1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train.toarray(), y_train.toarray(), test_size=0.2,
                                                        random_state=2)

plot_learning_curve(
    estimator=MlcLinReg(learning_rate=0.2,
                        iterations=1000,
                        batch_size=512,
                        l_one=0.15),
    title="Learning Curve",
    X=X_train2,
    y=y_train2,
    cv=5
)
np.set_printoptions(suppress=True)

mlc1 = MlcLinReg(learning_rate=0.1, iterations=500, batch_size=500, l_one=0.2)
mlc1.fit(X_train, y_train)
mlc1.plot_log_loss()
