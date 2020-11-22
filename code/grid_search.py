import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss= ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=8)

train_index, test_index = next(ss.split(X, y))

X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
scale.fit(X_train)

X_train =scale.transform(X_train)
X_test =scale.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.C = 1
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

C_range = [1e-5, 1e-3, 1e-2, 1, 1e2, 1e5, 1e10]

C_range_exp = np.arange(-15.0, 21.0)

C_range = 10 ** C_range_exp

from sklearn.model_selection import GridSearchCV

param = {'C': C_range}

gs = GridSearchCV(clf, param)
gs.fit(X_train, y_train)

clf_best = gs.best_estimator_

clf_best.fit(X_test, y_test)



plt.errorbar(gs.cv_results_['param_C'].data,
            gs.cv_results_['mean_fit_time'],
            yerr=gs.cv_results_['std_fit_time'], label='Training')

plt.errorbar(gs.cv_results_['param_C'].data,
            gs.cv_results_['mean_score_time'],yerr=gs.cv_results_['std_score_time'],
            label="tesv(val)")

plt.ylim(.6, 1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend(loc="best")
