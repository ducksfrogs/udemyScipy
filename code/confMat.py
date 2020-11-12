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
