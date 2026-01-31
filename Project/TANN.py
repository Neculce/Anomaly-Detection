import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
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

print("1. Loading Data...")
try:
    train_df = pd.read_csv(train_path, names=col_names)
    test_df = pd.read_csv(test_path, names=col_names)
    print("   Data loaded successfully!")
except FileNotFoundError:
    print("   ERROR: Files not found. Check the right sidebar 'Input' section.")

# --- 2. PREPARE LABELS & FEATURES ---
y_train_bin = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
y_test_bin = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

X_train = train_df.drop(['label', 'difficulty'], axis=1, errors='ignore')
X_test = test_df.drop(['label', 'difficulty'], axis=1, errors='ignore')

# --- 3. PREPROCESSING (With Error Fix) ---
print("2. Preprocessing (Scaling)...")
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Fit and Transform
X_train_vec = preprocessor.fit_transform(X_train)
X_test_vec = preprocessor.transform(X_test)

# --- THE FIX: Check if it's sparse before converting ---
if hasattr(X_train_vec, 'toarray'):
    X_train_vec = X_train_vec.toarray()
if hasattr(X_test_vec, 'toarray'):
    X_test_vec = X_test_vec.toarray()

print(f"   Original Features: {X_train_vec.shape[1]}")

# --- 4. TANN TRANSFORMATION ---
print("3. Generating Triangle Features (This takes a moment)...")

def generate_triangle_features(X):
    n, d = X.shape
    # Only calculate Upper Triangle to save memory/time
    r, c = np.triu_indices(d, k=1) 
    
    # Calculate product of feature pairs
    # Note: X must be a dense numpy array for this to work fast
    X_new = X[:, r] * X[:, c]
    return X_new

# Transform Train and Test
X_train_tann = generate_triangle_features(X_train_vec)
X_test_tann = generate_triangle_features(X_test_vec)

print(f"   TANN Features Generated: {X_train_tann.shape[1]}")

# --- 5. TRAIN & EVALUATE ---
k = 1
print(f"4. Training TANN k-NN (k={k})...")
knn_tann = KNeighborsClassifier(n_neighbors=k)
knn_tann.fit(X_train_tann, y_train_bin)

print("5. Predicting (Slower due to high dimensions)...")
y_pred = knn_tann.predict(X_test_tann)

# Metrics
acc = accuracy_score(y_test_bin, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
fpr = fp / (fp + tn)

print("\n" + "="*40)
print(f"TANN RESULTS (VS PAPER TABLE II)")
print("="*40)
print(f"Your Accuracy:      {acc * 100:.2f}%")
print(f"Paper Quote Acc:    96.91% [cite: 310, 311]")
print("-" * 40)
print(f"Your FPR:           {fpr * 100:.2f}%")
print(f"Paper Quote FPR:    3.83% [cite: 310]")
print("="*40)
