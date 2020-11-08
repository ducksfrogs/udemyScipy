import numpy as np


from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()


X = data.data


X.shape


X[0]


data.feature_names 


y = data.target



y


y.shape


data.DESCR


from sklearn import linear_model
clf = linear_model.LogisticRegression()



n_samples = X.shape[0]
n_train = n_samples //2 
n_test = n_samples - n_train



train_index = range(0, n_train)
test_index = range(n_train, n_samples)



X_train = X[train_index]
X_test = X[test_index]

y_train = y[train_index]
y_test = y[test_index]



X_train


train_index


np.array(train_index)
np.array(test_index)



clf.fit(X_train, y_train)



clf.score(X_train, y_train)



clf.score(X_test, y_test)


clf.predict(X_test)


wrong = 0 
for i, j in zip(clf.predict(X_test), y_test):
    if i == j:
        print (i, j)
    else:
        print (i, j, "Worng")
        wrong += 1




