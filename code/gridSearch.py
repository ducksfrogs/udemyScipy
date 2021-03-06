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


#plot error bar

plt.errorbar(gs.cv_results_['param_C'].data,
            gs.cv_results_['mean_strain_score'],
            yerr=gs.cv_results_['std_train_score'],
            label="training")

plt.errorbar(gs.cv_results_['param_C'].data,
            gs.cv_results_['mean_test_score'],
            yerr=gs.cv_results_['std_test_score'], label="test(val)")

plt.ylim(.6, 1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend(loc="best")


plt.errorbar(gs.cv_results_['param_C'].data,
            gs.cv_results_['mean_score_time'],
            yerr=gs.cv_results_['std_score_time'],
            label="test(val)")


plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt/plt.ylabel("computation time")
plt.legend(loc="best")




#SVC

from sklearn.svm import SVC

clf = SVC(gamma='auto')

C_range_exp = np.arange(-2.0, 5.0)
C_range = 10.0 ** C_range_exp

param = {'C': C_range,
        'kernel': ['linear', 'rbf']}

gs = GridSearchCV(clf, param, n_jobs=-1, verbose=2, return_train_score=True)

gs.best_params_,
gs.best_score_,
gs.best_estimator_

s_linear = (gs.cv_results_['param_kernel']=='linear').data

plt.plot(gs.cv_results_['param_C'][s_linear].data,
        gs.cv_results_['mean_train_score'][s_linear],
        label='training (linear)')


plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_test_score'][s_linear].data,
         label='test/val',
         linestyle='--')

s_rbf = (gs.cv_results_['param_kernel']=='rbf').data

plt.plot(gs.cv_results_['param_C'][s_rbf].data,
         gs.cv_results_['mean_test_score'][s_rbf],
         linestyle="--",
         label="test/val (rbf)")


gs.score(X_test, y_test)

SVC(kernel='rbf').gamma



from sklearn.svm import SVC

clf = SVC(gamma='auto')

C_range_exp = np.arange(-2.0, 10.0)
C_range = 10.0 ** C_range_exp

gamma_range_exp = np.arange(-10.0, 0.0, 3)
gamma_range = 10**gamma_range_exp

param = [{'C': C_range,
          'kernel':['linear']},
         {'C': C_range,
          'kernel':['rbf']}]

gs = GridSearchCV(clf, param, n_jobs=-1, verbose=2, return_train_score=True)
gs.fit(X_train, y_train)

gs.best_params_, gs.best_estimator_, gs.best_score_


s_linear = (gs.cv_results_['param_kernel']=='linear').data

plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_train_score'][s_linear],
         label='training (linear)')

plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_test_score'][s_linear],
         linestyle='--',
         label='test/val (linear)')

s_rbf = (gs.cv_results_['param_kernel'] == 'rbf').data

for g in gamma_range:
    s_gammma = (gs.cv_results_['param_gamma'][s_rbf].data==g)

    plt.plot(gs.cv_results_['param_C'][s_gammma].data,
             gs.cv_results_['mean_train_score'][s_rbf][s_gammma],
             label="training (rbf, gammma, {0:.e})".format(g))

    plt.plot(gs.cv_results_['param_C'][s_rbf][s_gammma],
            gs.cv_results_['mean_test_score'][s_rbf][s_gammma],
            linestyle='--',
            label='test/val (rbf, gammma, {0:.0e})'.format(g))

plt.ylim(.6, 1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


gs.score(X_test, y_test)



#KNN

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

param = {'n_neigbors': range(1,20)}

gs = GridSearchCV(clf, param, return_train_score=True)
gs.fit(X_train, y_train)

gs.best_params_, gs.best_score_, gs.best_estimator_

plt.errorbar(gs.cv_results_['param_n_neighbors'].data,
            gs.cv_results_['mean_train_score'],
            yerr=gs.cv_results_['std_train_score'],label='training')
