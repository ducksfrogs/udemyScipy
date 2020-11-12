import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import  load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit

clf = linear_model.LogisticRegression()


data = load_breast_cancer()


ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,test_size=0.5)


X = data.data
y = data.target

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train,y_test = y[train_index], y[test_index]


plt.scatter(data.data[:,3], data.data[:,4])
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])


plt.scatter(data.data[:,3], data.data[:,4])
plt.xlim(0,3000)
plt.ylim(0,3000)
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])


from sklearn.preprocessing import MinMaxScaler
mmsclaer = MinMaxScaler([-1,1])

mmsclaer.fit(X_train)

X_train_mmscale = mmsclaer.transform(X_train)

X_test_mmscale = mmscaler.transform(X_test)

plt.scatter(X_train_mmscale[:, 3],
             X_train_mmscale[:, 4],
             c='blue',
             label='train')
plt.scatter(X_test_mmscale[:, 3],
            X_test_mmscale[:, 4],c='red', label='test')
plt.xlabel(data.feature_names[3]+"(standaised)")
plt.ylabel(data.feature_names[3]+"(standaised)")
plt.legend(loc="best")

from sklearn import linear_model

clf = linear_model.LogisticRegression()
clf.fit(X_train_scale, y_train)
clf.score(X_test_scale, y_test)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
