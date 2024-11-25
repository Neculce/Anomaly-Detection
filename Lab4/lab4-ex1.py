import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Step 1: Generate Data (300 training samples, 200 test samples, 3 features, contamination=0.15)
n_train = 300
n_test = 200
n_features = 3
contamination = 0.15

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, 
                                                  n_test=n_test, 
                                                  n_features=n_features, 
                                                  contamination=contamination)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Step 2: Fit OC-SVM Model with Linear Kernel
ocsvm_linear = OCSVM(kernel='linear', contamination=contamination)
ocsvm_linear.fit(X_train)

# Predict labels using OC-SVM (linear kernel)
y_test_pred_linear = ocsvm_linear.predict(X_test)

# Compute Balanced Accuracy and ROC AUC for the linear kernel model
balanced_acc_linear = balanced_accuracy_score(y_test, y_test_pred_linear)
roc_auc_linear = roc_auc_score(y_test, ocsvm_linear.decision_function(X_test))

print(f"OC-SVM (Linear Kernel) - Balanced Accuracy: {balanced_acc_linear:.4f}")
print(f"OC-SVM (Linear Kernel) - ROC AUC: {roc_auc_linear:.4f}")

# Step 3: Visualize Training and Test Data (Ground Truth vs Predicted Labels for OC-SVM Linear)
fig = plt.figure(figsize=(14, 10))

# Plot ground truth labels
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm', label="Train Ground Truth")
ax1.set_title("Train Data - Ground Truth Labels")

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', label="Test Ground Truth")
ax2.set_title("Test Data - Ground Truth Labels")

# Plot predicted labels
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=ocsvm_linear.predict(X_train), cmap='coolwarm', label="Train Predicted Labels (Linear)")
ax3.set_title("Train Data - Predicted Labels (Linear)")

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_linear, cmap='coolwarm', label="Test Predicted Labels (Linear)")
ax4.set_title("Test Data - Predicted Labels (Linear)")

plt.tight_layout()
plt.show()

# Step 4: Fit OC-SVM Model with RBF Kernel
contamination = 0.20
ocsvm_rbf = OCSVM(kernel='rbf', contamination=contamination)
ocsvm_rbf.fit(X_train)

# Predict labels using OC-SVM (RBF kernel)
y_test_pred_rbf = ocsvm_rbf.predict(X_test)

# Compute Balanced Accuracy and ROC AUC for the RBF kernel model
balanced_acc_rbf = balanced_accuracy_score(y_test, y_test_pred_rbf)
roc_auc_rbf = roc_auc_score(y_test, ocsvm_rbf.decision_function(X_test))

print(f"OC-SVM (RBF Kernel) - Balanced Accuracy: {balanced_acc_rbf:.4f}")
print(f"OC-SVM (RBF Kernel) - ROC AUC: {roc_auc_rbf:.4f}")

# Step 5: Visualize Training and Test Data (Predicted Labels for OC-SVM RBF)
fig = plt.figure(figsize=(14, 10))

# Plot predicted labels for OC-SVM (RBF kernel)
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=ocsvm_rbf.predict(X_train), cmap='coolwarm', label="Train Predicted Labels (RBF)")
ax1.set_title("Train Data - Predicted Labels (RBF)")

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_rbf, cmap='coolwarm', label="Test Predicted Labels (RBF)")
ax2.set_title("Test Data - Predicted Labels (RBF)")

plt.tight_layout()
plt.show()

# Step 6: Fit DeepSVDD Model
deep_svdd = DeepSVDD(n_features=n_features, contamination=contamination)
deep_svdd.fit(X_train)

# Predict labels using DeepSVDD
y_test_pred_deep = deep_svdd.predict(X_test)

# Compute Balanced Accuracy and ROC AUC for DeepSVDD
balanced_acc_deep = balanced_accuracy_score(y_test, y_test_pred_deep)
roc_auc_deep = roc_auc_score(y_test, deep_svdd.decision_function(X_test))

print(f"DeepSVDD - Balanced Accuracy: {balanced_acc_deep:.4f}")
print(f"DeepSVDD - ROC AUC: {roc_auc_deep:.4f}")

# Step 7: Visualize Training and Test Data (Predicted Labels for DeepSVDD)
fig = plt.figure(figsize=(14, 10))

# Plot predicted labels for DeepSVDD
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=deep_svdd.predict(X_train), cmap='coolwarm', label="Train Predicted Labels (DeepSVDD)")
ax1.set_title("Train Data - Predicted Labels (DeepSVDD)")

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_deep, cmap='coolwarm', label="Test Predicted Labels (DeepSVDD)")
ax2.set_title("Test Data - Predicted Labels (DeepSVDD)")

plt.tight_layout()
plt.show()
