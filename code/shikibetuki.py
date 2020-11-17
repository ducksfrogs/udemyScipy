import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.set_cmap(plt.cm.gist_ncar)

X, y = make_blobs(n_samples=50,n_features=2, centers=5, cluster_std=0.8, random_state=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k')


def plotBoundary(X, clf, mesh=True, cmap=plt.get_cmap()):
    x_min = min(X[:, 0])
    x_max = max(X[:, 0])
    y_min = min(X[:, 1])
    y_max = max(X[:, 1])

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    z = z.reshape(XX.shape)

    if mesh:
        plt.pcolormesh(XX, YY, z, zorder=-10, cmap=cmap)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=1)

clf.fit(X, y)

plt.scatter(X[:,0], X[:,1], marker='o', edgecolors='k', s=50, c=y)
plotBoundary(X, clf)

X, y = make_blobs(n_samples=50,n_features=2, centers=5, cluster_std=2, random_state=3)
plt.scatter(X[:,0], X[:,1], c=y, s=50, edgecolors='k')

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)
plt.scatter(X[:,0], X[:,1], marker='o', edgecolors='k', s=50, c=y)
plotBoundary(X, clf)

for n in [1,5,10,15]:
    clf.n_neighbors = n
    plt.scatter(X[:, 0],X[:,1], marker='o',s=50, c=y, edgecolors='k')
    plotBoundary(X, clf)
    plt.title("{0}-NN".format(n))
    plt.show()
