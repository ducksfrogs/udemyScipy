import numpy as np
from sklearn.datasets import  load_iris
import matplotlib.pyplot as plt


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

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,
                  test_size=0.5,
                  train_size=0.5,
                  random_state=0)

train_index, test_index = next(ss.split(X))

list(test_index), list(test_index)

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

scores = []

for train_index, test_index in ss.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)


scores = np.array(scores)

print(scores)


scores.mean()
print("{0:4.2f} +- {1:4.2f} %".format(scores.mean()*100, scores.std()*100))

train_sizes = np.arange(0.1, 1, 0.1)

all_mean = []
all_std = []


for train_size in train_sizes:
    ss = ShuffleSplit(n_splits=100, test_size=1-train_size, train_size=train_size)
    scores = []
    for train_index, test_index in ss.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
    scores = np.array(scores)
    print("train_size {0:.0f}%: {1:4.2f} +- {2:4.2f}".format(train_size,scores.mean(),scores.std()))

    all_mean.append(scores.mean()*100)
    all_std.append(scores.std()*100)


plt.plot(train_sizes, all_mean)
plt.plot(train_sizes, all_mean)
plt.errorbar(train_sizes, all_mean, yerr=all_std)
