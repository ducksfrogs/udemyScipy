import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit

data = load_breast_cancer()

X = data.data
y = data.target

ss = ShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

conf_mat.sum()
conf_mat.diagonal().sum()
conf_mat.diagonal().sum() / conf_mat.sum()

TP = conf_mat[0,0]
TN = conf_mat[1,1]
FP = conf_mat[1,0]
FN = conf_mat[0,1]

TP, TN, FP, FN

from sklearn.metrics import  classification_report

print(classification_report(y_test, y_pred, digits=4))

recall_0 = TP / (TP + FN)

precision_0 = TP / (TP + FP)

recall_1 = TN / (FP + TN)

FP/ (TN+ FP)

precision_1 = TN / (TN + FN)

f1_score_0 = 2* recall_0 * precision_0 / (recall_0 + precision_0)
f1_score_1 = 2* recall_1 * precision_1 / (recall_1 + precision_1)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred, pos_label=0), f1_score(y_test, y_pred, pos_label=1)
