import numpy as np
import pandas as pd
from scipy.io import loadmat  # To load .mat files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

# Load the .mat file (replace with the actual file path)
mat_data = loadmat('cardio.mat')

# Inspect the keys to find data and labels
print(mat_data.keys())  # Check the structure of the .mat file

# Extract 'X' and 'y' from the loaded .mat file
X = mat_data['X']  # Features matrix (NumPy array)
y = mat_data['y'].flatten()  # Labels vector (flatten in case it's a column vector)

# Split into training (40%) and testing (60%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Assuming that the labels are in the form 0 (outlier) and 1 (inlier), we need to convert to
# OneClassSVM's expected format of -1 (outlier) and 1 (inlier)
y_train_svm = (y_train * 2) - 1  # Convert 0 -> -1, 1 -> 1
y_test_svm = (y_test * 2) - 1    # Convert 0 -> -1, 1 -> 1

# Define the parameter grid for GridSearchCV
param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],  # Only for kernels that use gamma
    'svm__nu': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create a pipeline with StandardScaler and OneClassSVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the data
    ('svm', OneClassSVM())  # OneClassSVM model
])

# Set up GridSearchCV with Balanced Accuracy as the scoring metric
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train_svm)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Convert predictions back from OneClassSVM format (-1, 1) to pyod format (0, 1)
y_pred_pyod = (-1*y_pred + 1) // 2  # Convert -1 -> 0, 1 -> 1

# Calculate the balanced accuracy score on the test set
balanced_accuracy = balanced_accuracy_score(y_test, y_pred_pyod)
print("Balanced Accuracy on test set:", balanced_accuracy)
print("With best model being: ", best_model)

