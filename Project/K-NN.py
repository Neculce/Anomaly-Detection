

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer

# --- KAGGLE CONFIGURATION ---
# Note: If your dataset is named differently in the 'Input' folder, 
# change 'nslkdd' below to match that folder name.
train_path = '/kaggle/input/nslkdd/KDDTrain+.txt'
test_path = '/kaggle/input/nslkdd/KDDTest+.txt'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"]

print("1. Loading Data...")
# We use try/except just in case the path is slightly different
try:
    train_df = pd.read_csv(train_path, names=col_names)
    test_df = pd.read_csv(test_path, names=col_names)
    print("   Data loaded successfully!")
except FileNotFoundError:
    print("   ERROR: Files not found. Check the right sidebar 'Input' section for the exact path.")

# Remove 'difficulty' column
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

# 2. Separate Features (X) and Target (y)
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 3. Handle Labels (0 = Normal, 1 = Attack)
print("2. Processing Labels...")
y_train_bin = y_train.apply(lambda x: 0 if x == 'normal' else 1)
y_test_bin = y_test.apply(lambda x: 0 if x == 'normal' else 1)

# 4. Preprocessing
print("3. Preprocessing Features (One-Hot + MinMax)...")
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Test different k values to see if we can hit 88% accuracy
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 29, 33, 39, 43]

print(f"{'k':<5} {'Accuracy':<10} {'FPR':<10}")
print("-" * 30)

for k in k_values:
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    knn_loop.fit(X_train_processed, y_train_bin)
    y_pred_loop = knn_loop.predict(X_test_processed)
    
    acc_loop = accuracy_score(y_test_bin, y_pred_loop)
    tn_loop, fp_loop, fn_loop, tp_loop = confusion_matrix(y_test_bin, y_pred_loop).ravel()
    fpr_loop = fp_loop / (fp_loop + tn_loop)
    
    print(f"{k:<5} {acc_loop*100:.2f}%     {fpr_loop*100:.2f}%")

# 5. k-NN Model
k = 5
print(f"4. Training k-NN (k={k})...")
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_processed, y_train_bin)

# 6. Prediction (This is the slow part!)
print("5. Predicting (This may take 1-2 minutes on Kaggle)...")
y_pred = knn.predict(X_test_processed)

# 7. Results
acc = accuracy_score(y_test_bin, y_pred)
conf_matrix = confusion_matrix(y_test_bin, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
fpr = fp / (fp + tn)

print("\n" + "="*30)
print(f"RESULTS VS PAPER (Table II)")
print("="*30)
print(f"Your Accuracy:      {acc * 100:.2f}%  (Paper: 88.91%)")
print(f"Your FPR:           {fpr * 100:.2f}%  (Paper: 38.02%)")
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test_bin, y_pred))

1. Loading Data...
   Data loaded successfully!
2. Processing Labels...
3. Preprocessing Features (One-Hot + MinMax)...
k     Accuracy   FPR       
------------------------------
1     77.92%     7.04%
3     77.16%     7.07%
5     76.94%     7.04%
7     76.87%     7.11%
9     76.87%     7.11%
11    76.83%     7.14%
13    77.38%     7.12%
15    77.45%     7.13%
17    77.48%     7.07%
19    77.52%     7.11%
25    77.59%     7.25%
29    77.60%     7.22%
33    77.67%     7.30%
39    77.78%     7.26%
43    76.95%     7.30%
4. Training k-NN (k=5)...
5. Predicting (This may take 1-2 minutes on Kaggle)...

==============================
RESULTS VS PAPER (Table II)
==============================
Your Accuracy:      76.94%  (Paper: 88.91%)
Your FPR:           7.04%  (Paper: 38.02%)
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.67      0.93      0.78      9711
           1       0.92      0.65      0.76     12833

    accuracy                           0.77     22544
   macro avg       0.80      0.79      0.77     22544
weighted avg       0.81      0.77      0.77     22544

from sklearn.metrics import roc_auc_score

# 1. Define the protocols we want to analyze
protocols = ['tcp', 'udp', 'icmp']

print("\n" + "="*40)
print(f"AUC SCORES BY PROTOCOL (Replicating Table I)")
print("="*40)
print(f"{'Protocol':<10} {'AUC Score':<10} {'Sample Size':<10}")
print("-" * 40)

# 2. Calculate AUC for each specific protocol
for proto in protocols:
    # A. Create a mask: True for rows that match the current protocol
    mask = test_df['protocol_type'] == proto
    
    # B. Filter the Processed Features and True Labels using the mask
    X_subset = X_test_processed[mask]
    y_subset = y_test_bin[mask]
    
    # C. Safety Check: We need both 'Normal' and 'Attack' samples to calculate AUC
    if len(np.unique(y_subset)) < 2:
        print(f"{proto:<10} {'Undefined (Only 1 class present)':<10} {len(y_subset):<10}")
        continue
        
    # D. Get Probabilities (AUC requires probabilities, not just 0/1 predictions)
    # predict_proba returns [prob_normal, prob_attack]. We want [:, 1]
    y_pred_proba = knn.predict_proba(X_subset)[:, 1]
    
    # E. Calculate Score
    auc = roc_auc_score(y_subset, y_pred_proba)
    print(f"{proto.upper():<10} {auc:.4f}     {len(y_subset):<10}")

# 3. Calculate AUC for "The Rest" (if any exist)
mask_rest = ~test_df['protocol_type'].isin(protocols)
if mask_rest.sum() > 0:
    X_subset = X_test_processed[mask_rest]
    y_subset = y_test_bin[mask_rest]
    
    if len(np.unique(y_subset)) < 2:
         print(f"{'OTHER':<10} {'Undefined':<10} {len(y_subset):<10}")
    else:
        y_pred_proba = knn.predict_proba(X_subset)[:, 1]
        auc = roc_auc_score(y_subset, y_pred_proba)
        print(f"{'OTHER':<10} {auc:.4f}     {len(y_subset):<10}")

print("-" * 40)
# 4. Overall AUC (Weighted Average)
y_prob_all = knn.predict_proba(X_test_processed)[:, 1]
auc_all = roc_auc_score(y_test_bin, y_prob_all)
print(f"{'OVERALL':<10} {auc_all:.4f}     {len(y_test_bin):<10}")

========================================
AUC SCORES BY PROTOCOL (Replicating Table I)
========================================
Protocol   AUC Score  Sample Size
----------------------------------------
TCP        0.8713     18880     
UDP        0.4560     2621      
ICMP       0.9180     1043      
----------------------------------------
OVERALL    0.8289     22544     


