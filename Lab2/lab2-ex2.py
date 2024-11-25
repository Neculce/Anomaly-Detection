# Using the function generate data clusters generate a 2-dimensional dataset with 400 train samples and 200 test samples that are organized in 2 clusters, with 0.1 contamination
#Train a KNN model from pyod.models.knn.
#  Use 4 subplots in order to display using different colors (for inliers and outliers): 
# • Ground truth labels for training data 
# • Predicted labels for training data 
# • Ground truth labels for test data 
# • Predicted labels for test data Use different values for the n neighbors parameter and observe how this affects the detection of small clusters of anomalies. Also compute the balanced accuracy for each parameter.


import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import pyod

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_features=2, contamination=0.1, random_state=42)

for i in range(1,17,4):
    knn = KNN(n_neighbors=i, contamination=0.1)
    knn.fit(X_train)

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Outliers', alpha=0.7)

    plt.title('Ground truth labels for training data')

    plt.subplot(2, 2, 2)
    predictions = knn.predict(X_train)
    plt.scatter(X_train[predictions == 0, 0], X_train[predictions == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X_train[predictions == 1, 0], X_train[predictions == 1, 1], c='red', label='Outliers', alpha=0.7)
    plt.title('Predicted labels for training data')

    plt.subplot(2, 2, 3)
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='red', label='Outliers', alpha=0.7)
    plt.title('Ground truth labels for test data')

    plt.subplot(2, 2, 4)
    predictions = knn.predict(X_test)
    plt.scatter(X_test[predictions == 0, 0], X_test[predictions == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X_test[predictions == 1, 0], X_test[predictions == 1, 1], c='red', label='Outliers', alpha=0.7)
    plt.title('Predicted labels for test data')

    print("Balanced accuracy (on test) for n_neighbors = " + str(i) + ": " + str(pyod.utils.data.evaluate_print('KNN', y_test, predictions)))
    print("Balanced accuracy (on train) for n_neighbors = " + str(i) + ": " + str(pyod.utils.data.evaluate_print('KNN', y_train, knn.predict(X_train))))
    plt.show()
