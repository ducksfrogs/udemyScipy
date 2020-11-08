import numpy as np
from sklearn.datasets import  load_iris

data = load_iris()
X = data.data
y = data.target



n_samples = X.shape[0]
n_train = n_samples //2 
n_test  = n_samples - n_train

train_index = range(0, n_train)
test_index = range(n_train, n_samples)



X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]



y


from sklearn import linear_model

clf = linear_model.LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)



clf.score(X_train, y_train)


y_train, y_test


from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,test_size=0.5, train_size=0.5,random_state=0)



train_index, test_index = next(ss.split(X))



ss.split(X_train)


X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]



clf.fit(X_train, y_train)


ss = ShuffleSplit(n_splits=10,
                  test_size=0.5, 
                  train_size=0.5, 
                  random_state=0)



next(ss.split(X))


scores = []

for train_index, test_index in ss.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)


scores = np.array(scores)

print(scores)



import matplotlib.pyplot as plt



train_sizes = np.arange(0.1, 1, 0.1)


train_sizes 




all_mean = []
all_std = []


for train_size in train_sizes:
    
    ss = ShuffleSplit(n_splits=100, train_size=train_size, test_size=1-train_size)
    scores = []
    for train_index, test_index in ss.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
    scores = np.array(scores)
    print("train_size {0:4.1f}get_ipython().run_line_magic(":", " {1:4.2f} +- {2:4.2f}\".format(train_size,scores.mean(),scores.std()))    ")







