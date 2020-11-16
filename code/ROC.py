import numpy as np

from sklearn.datasets import  load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,test_size=0.2, random_state=0)

train_index, test_index = next(ss.split(X,y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

clf = LogisticRegression()

clf.fit(X_train, y_train)


clf.score(X_test, y_test)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test, y_pred)

cmat = confusion_matrix(y_test, y_pred)

clf.decision_function(X_test[12:15])

clf.predict(X_test[12:15])

(clf.decision_function(X_test[12:15])>0).astype(int)


(clf.decision_function(X_test[12:15])>0.2).astype(int)
(clf.decision_function(X_test[12:15])>0).astype(int)

for th in range(-3, 7):
    print(th, (clf.decision_function(X_test[12:15]) > th).astype(int))


from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

test_score = clf.decision_function(X_test)

fpr, tpr, th = roc_curve(y_test, test_score)

plt.plot(fpr, tpr)
print("AUC = ", auc(fpr, tpr))

plt.plot([0,1], [0,1], linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylabel("True Positive Rate (recall)")
plt.xlabel("False Positive Rate (1-specificity)")


test_score = clf.decision_function(X_test)
precision, recall, th = precision_recall_curve(y_test, test_score)

plt.plot(recall, precision)

plt.xlim([-0.01, 1.01])
plt.ylim(0.0, 1.01)
plt.xlabel("Recall")
plt.ylabel("Precision")


test_score = clf.decision_function(X_test)
fpr, tpr = roc_curve(y_test, test_score)
plt.plot(fpr, tpr, label='result')
print("result AUC", auc(fpr, tpr))

test_score = np.random.uniform(size=y_test.size)
fpr, tpr = roc_curve(y_test, test_score)
plt.plot(fpr, tpr, label='random / chance')
print("chance AUC", auc(fpr, tpr))

fpr, tpr = roc_curve(y_test, y_test)
plt.plot(fpr, tpr, label="perfect")


test_score = clf.decision_function(X_test)
precision, recall, th = precision_recall_curve(y_test, test_score)
plt.plot(recall, precision, label="result")

test_score = np.random.uniform(size=y_test.size)
precision, recall, th = precision_recall_curve(y_test, test_score)
plt.plot(recall, precision, label='random')

precision, recall, th = precision_recall_curve(y_test, y_test)
plt.plot(recall, precision, label='perfect')

plt.legend(loc='best')
plt.xlim(-0.01, 1.01)
plt.ylim(0, 1.01)
