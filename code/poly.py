import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()


ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,test_size=0.2)


X = data.data
y = data.target

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train,y_test = y[train_index], y[test_index]


polf = PolynomialFeatures(degree=2)

polf.fit(X_train)

X_train_poly = polf.transform(X_train)
X_test_poly =polf.transform(X_test)

n_features = X_train_poly.shape[1]
i_range = range(1, n_features, 10)

X_train_poly_pca = pca.transform(X_train_poly)
X_test_poly_pca = pca.transform(X_test_poly)
