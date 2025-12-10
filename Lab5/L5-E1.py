import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Data Generation & Setup ---
np.random.seed(42) # For reproducibility

N = 500
mean_vector = [5, 10, 2]
cov_matrix = [[3, 2, 2], 
              [2, 10, 1], 
              [2, 1, 2]]

# Generate data
X = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=N)

# Plot Initial Data (3D)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6)
ax.set_title("Generated 3D Data")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
plt.show()

# --- PCA Implementation ---

# 1. Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 2. Compute Covariance Matrix
# Note: np.cov expects rows as variables, so we transpose X_centered.
# equivalent to (X_centered.T @ X_centered) / (N-1)
Sigma = np.cov(X_centered, rowvar=False) 

# 3. EVD (Eigenvalue Decomposition)
# eigh returns eigenvalues in ascending order
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

# Sort descending (lambda_max to lambda_min)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx] # Reorder columns to match eigenvalues

P = eigenvectors
Delta = np.diag(eigenvalues)

print(f"Eigenvalues: {eigenvalues}")

# --- 2. Variance Plotting ---

total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 5))
plt.bar(range(1, 4), explained_variance_ratio, alpha=0.5, align='center', label='Individual variance')
plt.step(range(1, 4), cumulative_variance, where='mid', label='Cumulative variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.title("Explained Variance by Component")
plt.show()

# --- 3. Outlier Detection: Single Component ---

# Project data into new space: X' = XP
X_prime = np.dot(X_centered, P)

def detect_single_component_outliers(component_index, X_transformed, contamination=0.1):
    """
    Detects outliers based on deviation along a specific PC index (0-based).
    """
    # Extract the specific component (column)
    component_data = X_transformed[:, component_index]
    
    # Deviation from mean (mean is 0 for centered data)
    # We look at the absolute deviation
    deviations = np.abs(component_data)
    
    # Calculate threshold using quantile
    threshold = np.quantile(deviations, 1 - contamination)
    
    # Create labels: 1 for outlier, 0 for normal
    labels = deviations > threshold
    return labels

# Task: Detect on 3rd PC (index 2) and 2nd PC (index 1)
for pc_idx in [2, 1]:
    labels = detect_single_component_outliers(pc_idx, X_prime, contamination=0.1)
    
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot normal points
    ax.scatter(X[~labels, 0], X[~labels, 1], X[~labels, 2], c='b', alpha=0.5, label='Normal')
    # Plot outliers
    ax.scatter(X[labels, 0], X[labels, 1], X[labels, 2], c='r', marker='^', s=50, label='Anomaly')
    
    ax.set_title(f"Outliers detected via PC {pc_idx + 1}")
    ax.legend()
    plt.show()

# --- 4. Outlier Detection: Normalized Distance (Mahalanobis) ---

# Algorithm steps from lab sheet:
# 1. EVD (Done)
# 2. Transform X' = XP (Done, X_prime)
# 3. Normalize X' = X' * Delta^(-1/2)
Delta_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues))
X_normalized = np.dot(X_prime, Delta_inv_sqrt)

# 4. Anomaly score: squared Euclidean distance
anomaly_scores = np.sum(X_normalized**2, axis=1)

# Thresholding
contamination = 0.1
threshold_mahalanobis = np.quantile(anomaly_scores, 1 - contamination)
labels_mahalanobis = anomaly_scores > threshold_mahalanobis

# Plotting Final Result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[~labels_mahalanobis, 0], X[~labels_mahalanobis, 1], X[~labels_mahalanobis, 2], 
           c='b', alpha=0.5, label='Normal')
ax.scatter(X[labels_mahalanobis, 0], X[labels_mahalanobis, 1], X[labels_mahalanobis, 2], 
           c='r', marker='^', s=50, label='Anomaly')
ax.set_title("Outliers detected via Normalized Distance (Mahalanobis)")
ax.legend()
plt.show()
