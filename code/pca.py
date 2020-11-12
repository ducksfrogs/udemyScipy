import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

import seaborn as sns

from sklearn.decomposition import PCA

X = data.data[:, [0,2]]
y = data.target
names = data.feature_names[[0,2]]

pca = PCA()

pca.fit(X)

plt.scatter(X[:, 0], X[:, 1])
plt.xlim(0,100)
plt.ylim(20,200)
plt.xlabel(names[0])
plt.ylabel(names[1])

pca.explained_variance_



X = data.data[:, [6,7]]
y = data.target

names = data.feature_names[[6,7]]

plt.scatter(X[:,0], X[:, 1])
plt.xlim(0,0.5)
plt.ylim(0, 0.5)
plt.xlabel(names[0])
plt.ylabel(names[1])


X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8, random_state=0)
train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

pca.fit(X_train)

plt.plot(pca.explained_variance_ratio_)

plt.plot(np.add.accumulate(pca.explained_variance_ratio_))


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


scores = []
i_range = range(1,31)

for i in i_range:

    clf.fit(X_train_pca[:, 0:i], y_train)
    scores.append(clf.score(X_test_pca[: 0:i], y_test))

scores = np.array(scores)

plt.plot
