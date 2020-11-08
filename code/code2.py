import numpy as np
from sklearn.datasets import  load_iris

data = load_iris()
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,test_size=0.5, train_size=0.5,random_state=0)
n_samples = X.shape[0]
n_train = n_samples //2
n_test  = n_samples - n_train

train_index = range(0, n_train)
test_index = range(n_train, n_samples)

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

from sklearn import linear_model

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
