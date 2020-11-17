import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=20,
                           n_features=2,
                           n_classes=3,
                           n_clusters_per_class=1,
                           n_redundant=0,
                            n_repeated=0,
                            random_state=8)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50)


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

plt.scatter(X[:,0], X[:, 1], marker='o', s=50, c=y, edgecolor='k')

from sklearn.linear_model import  LogisticRegression

clf = LogisticRegression()
clf.fit(X, y)

plt.scatter(X[:,0], X[:, 1], marker='o', s=50, c=y)
plotBoundary(X, clf)

from sklearn import svm

clf = svm.SVC(kernel='linear', C=10)

clf.fit(X,y)

plt.scatter(X[:,0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary(X, clf)

clf = svm.SVC(kernel='rbf', C=10)

clf.fit(X,y)

plt.scatter(X[:,0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary(X, clf)


def plotBoundary2(X, clf, boundary=True):
    colors = ['k'];
    linestyle = ['-'];
    levels = [0]

    x_min = min(X[:, 0])
    x_max = max(X[:, 0])
    y_min = min(X[:, 1])
    x_max = max(X[:, 1])

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
