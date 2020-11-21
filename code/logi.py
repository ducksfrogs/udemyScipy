import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8, random_state=0)

train_index, test_index = next(ss.split(X,y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf.C = 1e-3
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


X_test_value = clf.decision_function(X_test)

sorted_va  = np.sort(X_test_value)

plt.plot(X_test_value)
plt.plot([0,120],[0,0], linestyle='--')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.plot(sigmoid(sorted_va))
plt.plot([0,120], [0.5, 0.5], linestyle='--')
