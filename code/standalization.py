import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import  load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit

clf = linear_model.LogisticRegression()


data = load_breast_cancer()

X = data.data
y = data.target


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
