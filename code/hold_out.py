import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

minsit = fetch_openml('mnist_784', version=1)
minsit.COL_NAMES

X_train = minsit.data[:60000]
X_test = minsit.data[60000:70000]

y_train = minsit.target[:60000]
y_test = minsit.target[60000:70000]

from sklearn.datasets import  load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit

clf = linear_model.LogisticRegression()

ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,test_size=0.5)

data = load_breast_cancer()

X = data.data
y = data.target

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train,y_test = y[train_index], y[test_index]

ss = ShuffleSplit(n_splits=10,
                  train_size=0.5,test_size=0.5)



print(np.unique(y, return_counts=True))
for train_index,test_index in ss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import KFold

ss = KFold(n_splits=10, shuffle=True)

for train_index, test_index in ss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(np.unique(y_train, return_counts=True)[1]/y_train.size,y_train.size,
          np.unique(y_test, return_counts=True)[1]/ y_test.size)

from sklearn.model_selection import StratifiedKFold
ss = StratifiedKFold(n_splits=10, shuffle=True)

for train_index, test_index in ss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(np.unique(y_train, return_counts=True)[1]/y_train.size,y_train.size,
          np.unique(y_test, return_counts=True)[1]/ y_test.size, y_test.size)


from sklearn.model_selection import cross_val_score

ave_score = cross_val_score(clf, X, y, cv=10)
