import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

# 1D Dataset, 10% contamination
n_samples = 500
contamination_rate = 0.1  # 10% contamination

X_train, y_train = generate_data(n_train=n_samples, n_test=0, n_features=2, contamination=contamination_rate)

# Z-Score computation for a 1D Dataset
z_scores = (X_train - np.mean(X_train)) / np.std(X_train)

# Calculation the threshold
threshold = np.quantile(np.abs(z_scores), 1 - contamination_rate)

# Based on threshold we classify points as outliers if their threshold is too big
y_train_pred = (np.abs(z_scores) > threshold).astype(int).ravel()

# Computation of the confusion matrix and balanced accuracy
cm = confusion_matrix(y_train, y_train_pred)
balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

# Plot the dataset with detected anomalies
plt.figure(figsize=(8, 6))
plt.scatter(np.arange(len(X_train)), X_train, c='blue', label='Normal Data')
plt.scatter(np.arange(len(X_train))[y_train == 1], X_train[y_train == 1], c='red', label='True Outliers')
plt.scatter(np.arange(len(X_train))[y_train_pred == 1], X_train[y_train_pred == 1], facecolors='none', edgecolors='green', label='Detected Outliers')
plt.title('Unidimensional Dataset with True and Detected Anomalies (Z-score method)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Print confusion matrix and balanced accuracy
print(f"Confusion Matrix:\n{cm}")
print(f"Balanced Accuracy: {balanced_acc:.2f}")

# Return Z-score threshold for reference
print(f"Z-score threshold: {threshold}")
