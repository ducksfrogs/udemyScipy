import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

from sklearn.model_selection import train_test_split

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=8, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear')


C_range = [1e-5, 1e-3, 1e-2, 1, 1e2, 1e5, 1e10]

C_range_exp = np.arange(-15, 21)
C_range = 10 ** C_range_exp

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore',ConvergenceWarning)

from sklearn.model_selection import GridSearchCV

param = {'C':C_range}

gs = GridSearchCV(clf, param, return_train_score=True)
gs.fit(X_train, y_train)

clf_best = gs.best_estimator_
clf_best.score(X_test, y_test)

gs.score(X_test, y_test)
