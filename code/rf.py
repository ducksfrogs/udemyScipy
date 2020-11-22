import numpy as np
import matplotlib.pyplot as plt

plt.set_cmap(plt.cm.Paired)

def plotBoundary(X, clf, mesh=True, boundary=True, type='predict'):

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid

    if type == 'predict':
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    elif type == 'probability':
        Z = np.exp(clf.predict_log_proba(np.c_[XX.ravel(), YY.ravel()]))[:, 1]
    else:
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    Z = Z.reshape(XX.shape) # just reshape

    if mesh:
        if type == 'predict':
            plt.pcolormesh(XX, YY, Z, zorder=-10) # paint in 2 colors, if Z > 0 or not
        else:
            plt.pcolormesh(XX, YY, Z, zorder=-10, cmap=plt.cm.bwr)
            plt.colorbar()

    if boundary:
        level = [0.5]
        if type == "probability":
            level = [0.5]
        plt.contour(XX, YY, Z,
                    colors='k', linestyles='-', levels=level)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=2, random_state=3)

plt.scatter(X[:,0], X[:,1], c=y, s=50, edgecolors='k')

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=8)
clf.max_depth = 1
clf.n_estimators = 1
clf.fit(X,y)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plotBoundary(X, clf)

for i in range(3,10):
    clf.n_estimators = i
    clf.fit(X, y)
    plt.scatter(X[:,0], X[:,1], marker='o', c=y)
    plotBoundary(X, clf)
    plt.title("{0} estimators".format(i))
    plt.show()
