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

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

clf.score(X_train, y_train)

clf.score(X_test, y_test)


 n_range = range(1,20)
 scores = []

 for n in n_range:
     clf.n_neighbors = n
     score = clf.score(X_test, y_test)
     print(n ,score)
     scores.append(score)
scores = np.array(scores)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)

X_test_scale = scaler.transform(X_test)

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
scores2 []
clf.fit(X_train_scale, y_train)
for n in n_range:
    clf.n_neighbors = n
    score = clf.score(X_test_scale, y_test)
    print(n, score)
    scores.append(score)

scores2 = np.array(scores2)

plt.plot(n_range, scores)
plt.plot(n_range, scores2)
