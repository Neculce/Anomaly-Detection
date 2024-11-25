import numpy as np
from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generation of a 2D dataset
np.random.seed(42)

# Parameters for normal data
mean = [0, 0]  # Mean of the normal data
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix 
n_samples = 1000  # Number of samples
contamination_rate = 0.1  # 10% contamination rate as usual

# Generation of normal data
X_normal = np.random.multivariate_normal(mean, cov, int(n_samples * (1 - contamination_rate)))

# Generation of outliers (anomalies) with a different distribution
X_outliers = np.random.uniform(low=-6, high=6, size=(int(n_samples * contamination_rate), 2))

# Combine normal data with outliers
X_train = np.vstack([X_normal, X_outliers])

# True labels (0 for normal, 1 for outliers)
y_train = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_outliers))])

# Z-Score computation for each feature
z_scores = zscore(X_train, axis=0)

# Calculation of the Z-Score threshold
threshold = np.quantile(np.abs(z_scores), 1 - contamination_rate)

# Data points classification based on Z-Score threshold
y_train_pred = (np.abs(z_scores) > threshold).any(axis=1).astype(int)

# Computation of confusion matrix and balanced accuracy scores
cm = confusion_matrix(y_train, y_train_pred)
balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

# Data plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Normal')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='True Outliers')
plt.scatter(X_train[y_train_pred == 1][:, 0], X_train[y_train_pred == 1][:, 1], facecolors='none', edgecolors='green', label='Detected Outliers')
plt.title('2D Dataset with True and Detected Anomalies (Z-score method)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Print confusion matrix and balanced accuracy
print(f"Confusion Matrix:\n{cm}")
print(f"Balanced Accuracy: {balanced_acc:.2f}")

# Returning Z-Score and balanced accuracy for refference
threshold, balanced_acc
