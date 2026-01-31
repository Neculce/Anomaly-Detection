import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer

# --- 1. LOAD DATA ---
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

try:
    train_df = pd.read_csv(train_path, names=col_names)
    test_df = pd.read_csv(test_path, names=col_names)
except FileNotFoundError:
    print("Error: Fix the file path!")

# Drop difficulty
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

# Create Binary Labels (0=Normal, 1=Attack)
y_train_all = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
y_test = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# --- 2. SEPARATE NORMAL TRAFFIC FOR TRAINING ---
# PCA Anomaly Detection is trained ONLY on Normal data
train_normal_df = train_df[train_df['label'] == 'normal']
X_train_normal = train_normal_df.drop('label', axis=1)

# We test on EVERYTHING (Normal + Attack)
X_test = test_df.drop('label', axis=1)

# --- 3. PREPROCESSING ---
print("Preprocessing...")
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [c for c in X_train_normal.columns if c not in categorical_cols]

# Important: Fit the scaler ONLY on Normal Training Data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train_normal)
X_test_processed = preprocessor.transform(X_test)

# --- 4. PCA IMPLEMENTATION ---
# The paper specifies 16 Principal Components (Page 4, Sec A)
n_components = 24
print(f"Training PCA with {n_components} components...")

pca = PCA(n_components=n_components, random_state=42)
pca.fit(X_train_processed)

# --- 5. CALCULATE RECONSTRUCTION ERROR ---
# Logic: Compress -> Decompress. Measure the loss.
print("Calculating Reconstruction Error...")

# Transform (Encode) and Inverse Transform (Decode)
X_test_pca = pca.transform(X_test_processed)
X_test_reconstructed = pca.inverse_transform(X_test_pca)

# Calculate Mean Squared Error (MSE) between Input and Output
# This 'mse' is our Anomaly Score. High MSE = Anomaly.
reconstruction_error = np.mean(np.square(X_test_processed - X_test_reconstructed), axis=1)

# --- 6. EVALUATION (Replicating Table I) ---
print("\n" + "="*40)
print(f"PCA RESULTS (VS PAPER TABLE I)")
print("="*40)

protocols = ['tcp', 'udp', 'icmp']
overall_auc = roc_auc_score(y_test, reconstruction_error)

print(f"{'Protocol':<10} {'My AUC':<10} {'Paper AUC':<10} {'Status':<10}")
print("-" * 50)

for proto in protocols:
    mask = test_df['protocol_type'] == proto
    if mask.sum() > 0:
        y_subset = y_test[mask]
        score_subset = reconstruction_error[mask]
        
        if len(np.unique(y_subset)) > 1:
            auc = roc_auc_score(y_subset, score_subset)
            
            # Paper Values from Table I [cite: 240]
            paper_val = "N/A"
            if proto == 'tcp': paper_val = "0.9360"
            elif proto == 'udp': paper_val = "0.8937"
            elif proto == 'icmp': paper_val = "0.8941"
            
            status = "BEAT IT" if auc > float(paper_val) else "LOST"
            print(f"{proto.upper():<10} {auc:.4f}     {paper_val:<10} {status}")

print("-" * 50)
print(f"{'OVERALL':<10} {overall_auc:.4f}     {'0.9292':<10} {'BEAT IT' if overall_auc > 0.9292 else 'LOST'}")
