import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()


data.feature_names


df = pd.DataFrame(data.data[:, 0:10], columns=data.feature_names[0:10])


len(data.feature_names)


data.data[:, 0:10]


df


from pandas.plotting import scatter_matrix


scatter_matrix (df, figsize=(10,10))


df = pd.DataFrame(data.data[:, 6:8], columns=data.feature_names[6:8])
scatter_matrix(df, figsize=(3,3))


df = pd.DataFrame(data.data[:, [0,2]], columns=data.feature_names[[0,2]])


scatter_matrix(df, figsize=(4,4))


X = data.data[:, [0,2]]
y = data.target
names = data.feature_names[[0,2]]


X.shape 


plt.scatter(X[:, 0], X[:,1])
plt.xlim(0,180)
plt.ylim(20,200)
plt.xlabel(names[0])
plt.ylabel(names[1])


from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X)



X_new = pca.transform(X)


plt.scatter(X_new[:, 0], X_new[: ,1])
plt.ylim(-20,200)
plt.xlabel(names[0])
plt.ylabel(names[1])



pca.explained_variance_


pca.explained_variance_ / pca.explained_variance_.sum()


pca.explained_variance_ratio_


X = data.data[:, [6,7]]
y = data.target

names = data.feature_names[[6,7]]

plt.scatter(X[:,0], X[:, 1])
plt.xlim(0,0.5)
plt.ylim(0, 0.5)
plt.xlabel(names[0])
plt.ylabel(names[1])



pca = PCA()

pca.fit(X)



X_new = pca.transform(X)


plt.scatter(X_new[:, 0], X_new[: ,1])
plt.ylim(-20,200)
plt.xlabel(names[0])
plt.ylabel(names[1])



m = X.mean(axis=0)


Xp = (X - m)


c = Xp.transpose().dot(Xp)


w, _ = np.linalg.eig(c)


w


c


w / w.sum()


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



X_train_pca


from sklearn import linear_model
clf = linear_model.LogisticRegression()


clf.fit(X_train_pca, y_train)


clf.score(X_test_pca, y_test)


clf.fit(X_train_pca[:, 0:1], y_train)
clf.score(X_test_pca[:, 0:1], y_test)


clf.fit(X_train_pca[:, 0:3], y_train)
clf.score(X_test_pca[:, 0:3], y_test)


scores = []
i_range = range(1,31)

for i in i_range:

    clf.fit(X_train_pca[:, 0:i], y_train)
    scores.append(clf.score(X_test_pca[:, 0:i], y_test))

scores = np.array(scores)



plt.plot(i_range, scores)


for i in range(1,31):
    print(i)



