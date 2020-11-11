import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 500

X = np.random.uniform(low=0, high=1, size=[N,2])
y = np.random.choice([0,1], size=N)

plt.scatter(X[:, 0], X[:,1], c=y, s=50)

from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=1)

X_train = X
X_test = X

y_train = y
y_test = y

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=10)
