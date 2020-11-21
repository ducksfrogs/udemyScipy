import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression


X, y = make_blobs(n_samples=20,n_features=2,
                  centers=2, random_state=3)


plt.scatter(X[:, 0], X[:, 1],c=y, s=50)
for l, dx, dy in zip(y, x[:,0], x[:,1]):
    plt.annotate(l, xy=(dx-0.2, dy+0.4))

def plotBoundary(X, clf, mksh=True, boundary=True, )

clf = LogisticRegression()

clf.fit(X, y)
