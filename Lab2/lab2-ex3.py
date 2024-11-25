
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import pyod
import sklearn
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF

cluster_1, _ = make_blobs(n_samples=200, n_features=2, centers=1, cluster_std=2, center_box=(-10, -10))
cluster_2, _ = make_blobs(n_samples=100, n_features=2, centers=1, cluster_std=6, center_box=(10, 10))

X = np.vstack((cluster_1, cluster_2))

knn = KNN(contamination=0.07, n_neighbors=20)
lof = LOF(contamination=0.07, n_neighbors=20)

y_knn_pred = knn.fit_predict(X)
y_lof_pred = lof.fit_predict(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("KNN vs LOF on Clusters with Different Densities")

ax1.scatter(X[:, 0], X[:, 1], c=y_knn_pred, cmap='coolwarm', marker='o', edgecolor='k')
ax1.set_title("KNN (n_neighbors=20)")

ax2.scatter(X[:, 0], X[:, 1], c=y_lof_pred, cmap='coolwarm', marker='o', edgecolor='k')
ax2.set_title("LOF (n_neighbors=20)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
