import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

# Step 1: Generate the data
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,  # 400 training samples
    n_test=100,   # 100 test samples
    n_features=2,  # 2-dimensional dataset
    contamination=0.1,  # 10% contamination (outliers)
    random_state=42  # For reproducibility
)

#plot data

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
            c='blue', label='Inliers', alpha=0.7)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
            c='red', label='Outliers', alpha=0.7)

#prep and show data

plt.title('Scatter Plot of Training Data (Inliers vs Outliers)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()