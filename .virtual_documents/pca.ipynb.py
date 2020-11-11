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
scatter_matrix(df, figsize=(5,5))



