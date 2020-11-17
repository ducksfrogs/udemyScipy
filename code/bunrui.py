import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=20,
                           n_features=2,
                           n_classes=3,
                           n_clusters_per_class=1,n_redundant=0, n_repeated=0, random_state=8)
