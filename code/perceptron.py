import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=1, random_state=8)

plt.scatter(X[:,0], X[:,1], c=y, s=50)


def plotBoundary(X, clf, mesh=True, boundary=True, cmap=plt.get_cmap()):
    x_min = min(X[:, 0])
    x_max = max(X[:, 0])
    y_min = min(X[:, 1])
    y_max = max(X[:, 1])

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    z = z.reshape(XX.shape)

    if mesh:
        plt.pcolormesh(XX, YY, z, zorder=-10, cmap=cmap)
    if boundary:
        plt.contourf(XX, YY, z, colors='k', linestyle='--' )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


from sklearn.linear_model import Perceptron

clf = Perceptron()

clf.fit(X, y)

plt.scatter(X[:,0], X[:,1], marker='o', c=y, edgecolors='k')
plotBoundary(X, clf)


X, y = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=1, random_state=8)

plt.scatter(X[:,0], X[:,1], c=y, s=50)

for s in range(10):
    clf.random_state = s
    clf.fit(X, y)
    plotBoundary(X, clf, mesh=False)

plt.scatter(X[:,0], X[:,1], marker='o', c=y, s=50, edgecolors='k')
