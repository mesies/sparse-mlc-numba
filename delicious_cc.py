from MlcClassifierChains import MlcClassifierChains
from MlcScore import score_accuracy
from helpers import *

profile = lambda none: none
# '@profile' is used by line_profiler but the python interpreter does not recognise the decorator, so in order to edit
# as few lines as possible each time line_profiler is run, a lambda is used.
# Comment when debugging with line profiler

ti = tic()

# X_train_, y_train_, X_test, y_test = load_delicious()
X_train_, y_train_, X_test, y_test = load_mlc_dataset("delicious_large")
X_train, y_train = shuffle_dataset(X_train_, y_train_)

mlc = MlcClassifierChains(learning_rate=1.2, iterations=50, batch_size=320)
mlc.fit(X_train, y_train)

y_pred = mlc.predict(X_test)

print score_accuracy(y_pred, y_test)