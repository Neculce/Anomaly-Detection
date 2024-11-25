# Import necessary libraries
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt

# 2D Dataset, 10% contamination, 400 training points, 100 test points
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

# Ploting data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Normal')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Outliers')
plt.title('2D Dataset - Training Samples with Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Intialization of the KNN model
contamination_rate = 0.1
knn = KNN(contamination=contamination_rate)
knn.fit(X_train)

# Predictions for training
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Decision scores for the ROC curve computation
y_train_scores = knn.decision_function(X_train)
y_test_scores = knn.decision_function(X_test)

# Confusion matrix calculation and TN FP FN TP extraction
cm_train = confusion_matrix(y_train, y_train_pred)
tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

# Computing balance accuracy 
balanced_acc_train = balanced_accuracy_score(y_train, y_train_pred)

# Plot of the ROC curve and general plot.
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (contamination={contamination_rate})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN Model')
plt.legend()
plt.grid()
plt.show()

# Conf. Matrix printing
print(f"Confusion Matrix (Training Data):\nTN: {tn_train}, FP: {fp_train}, FN: {fn_train}, TP: {tp_train}")
print(f"Balanced Accuracy (Training Data): {balanced_acc_train:.2f}")

# The following code is the same except the contamination rate has been changed to 20%
new_contamination_rate = 0.2 
knn_new = KNN(contamination=new_contamination_rate)
knn_new.fit(X_train)

y_train_pred_new = knn_new.predict(X_train)
y_test_pred_new = knn_new.predict(X_test)
y_test_scores_new = knn_new.decision_function(X_test)

cm_train_new = confusion_matrix(y_train, y_train_pred_new)
tn_train_new, fp_train_new, fn_train_new, tp_train_new = cm_train_new.ravel()
balanced_acc_train_new = balanced_accuracy_score(y_train, y_train_pred_new)
fpr_new, tpr_new, thresholds_new = roc_curve(y_test, y_test_scores_new)

plt.figure(figsize=(8, 6))
plt.plot(fpr_new, tpr_new, label=f'ROC Curve (contamination={new_contamination_rate})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN Model with New Contamination Rate')
plt.legend()
plt.grid()
plt.show()

# Print updated confusion matrix and balanced accuracy
print(f"Confusion Matrix (Training Data with new contamination rate):\nTN: {tn_train_new}, FP: {fp_train_new}, FN: {fn_train_new}, TP: {tp_train_new}")
print(f"Balanced Accuracy (Training Data with new contamination rate): {balanced_acc_train_new:.2f}")
