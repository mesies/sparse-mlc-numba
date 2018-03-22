from sklearn.model_selection import cross_val_score

import helpers
from MlcLinReg import MlcLinReg

X_train, y_train, X_test, y_test = helpers.load_delicious(2)
helpers.plot_confusion_matrix(MlcLinReg(learning_rate=0.5,
                                        iterations=2000,
                                        batch_size=1,
                                        l_one=0.1)
                              , X=X_train.toarray(), y=y_train.toarray())
print cross_val_score(estimator=MlcLinReg(learning_rate=0.5,
                                          iterations=2000,
                                          batch_size=1,
                                          l_one=0.1),
                      X=X_train.toarray(),
                      y=y_train.toarray(), cv=20).mean()
