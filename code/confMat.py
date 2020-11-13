import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()



X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8, random_state=0)
train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

from sklearn import linear_model
clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

np.count_nonzero(y_test==0), np.count_nonzero(y_test==1)

y_pred = clf.predict(X_test)

conf_mat = np.zeros([2,2])

for true_label, est_label in zip(y_test, y_pred):
    conf_mat[true_label, est_label] +=1

print(conf_mat)

df = pd.DataFrame(conf_mat, columns=['pred 0', 'pred 1'], index=['true 0', 'true 1'])

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(y_test, y_pred)

TP = cmat[0,0]
TN = cmat[1,1]

FN = cmat[0,1]
FP = cmat[1,0]

from sklearn.datasets import load_digits

data = load_digits()

X = data.data
y= data.target

img = data.images

for i in range(10):
    i_th_digit = data.images[data.target==i]
    for j in range(0,15):
        plt.subplot(10, 15, i * 15 + j +1)
        plt.axis('off')
        plt.imshow(i_th_digit[j], interpolation='none')

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

clf.fit(X_train, y_train)
clf.score(X_test, y_test)


y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)

df = pd.DataFrame(conf_mat,
                  columns=range(0,10),
                  index=range(0,10))

from sklearn.decomposition import PCA
pca = PCA(whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf.fit(X_train_pca, y_train)
clf.score(X_test_pca, y_test)

for i in range(10):
    i_th_digit = X_train_pca[y_train==i]
    for j in range(0,15):
        plt.subplot(10, 15, i * 15 + j  +1)
        plt.axis('off')
        plt.imshow(i_th_digit[j].reshape(8,8), interpolation=None)


y_pred_pca = clf.predict(X_test_pca)

conf_mat = confusion_matrix(y_test, y_pred_pca)

df = pd.DataFrame(conf_mat,
                  columns=range(0,10),
                  index=range(0,10))



scores = []

for i in range(1,65):
    clf.fit(X_train_pca[:, 0:i], y_train)
    score = clf.score(X_test_pca[:, 0:i], y_test)
    print(i, score)
    scores.append(score)

scores = np.array(scores)

plt.plot(scores)
plt.ylim(0.9, 1)
