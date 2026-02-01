"""
SVM with GridSearchCV for Network Anomaly Detection on NSL-KDD Dataset
Finds optimal hyperparameters automatically
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# NSL-KDD column names (41 features + 2 labels)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL = ['protocol_type', 'service', 'flag']


def load_data(train_path, test_path=None):
    df_train = pd.read_csv(train_path, names=COLUMNS)
    if test_path:
        df_test = pd.read_csv(test_path, names=COLUMNS)
        return df_train, df_test
    return df_train, None


def preprocess(df_train, df_test=None):
    df_train['binary_label'] = (df_train['label'] != 'normal').astype(int)
    if df_test is not None:
        df_test['binary_label'] = (df_test['label'] != 'normal').astype(int)
    
    for col in CATEGORICAL:
        le = LabelEncoder()
        if df_test is not None:
            le.fit(pd.concat([df_train[col], df_test[col]]))
        else:
            le.fit(df_train[col])
        df_train[col] = le.transform(df_train[col])
        if df_test is not None:
            df_test[col] = le.transform(df_test[col])
    
    feature_cols = [c for c in COLUMNS if c not in ['label', 'difficulty']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['binary_label'].values
    
    if df_test is not None:
        X_test = df_test[feature_cols].values
        y_test = df_test['binary_label'].values
    else:
        X_test, y_test = None, None
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


def grid_search_svm(X_train, y_train):
    """Find best SVM parameters using GridSearchCV."""
    
    # Parameter grid to search
    param_grid = [
        # RBF kernel
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        },
        # Linear kernel
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100]
        },
        # Polynomial kernel
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]
        }
    ]
    
    svm = SVC(random_state=42)
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=2
    )
    
    print("Starting GridSearchCV...")
    print(f"Testing {sum(len(pd.ParameterGrid(p)) for p in param_grid)} parameter combinations\n")
    
    grid_search.fit(X_train, y_train)
    
    return grid_search


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    accuracy = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    
    print(f"Detection Accuracy: {accuracy * 100:.2f}%")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print(f"True Positive Rate: {tpr * 100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")


def main():
    # Load NSL-KDD dataset
    df_train, df_test = load_data('KDDTrain+.txt', 'KDDTest+.txt')
    X_train, y_train, X_test, y_test = preprocess(df_train, df_test)
    print(f"Loaded NSL-KDD: {len(X_train)} train, {len(X_test)} test samples\n")
    
    # Run grid search
    grid_search = grid_search_svm(X_train, y_train)
    
    # Print results
    print("\n" + "="*50)
    print("GRID SEARCH RESULTS")
    print("="*50)
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # Show top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values('rank_test_score')
    for i, row in results.head(5).iterrows():
        print(f"  {row['rank_test_score']}. {row['params']} -> {row['mean_test_score']*100:.2f}%")
    
    # Evaluate best model on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION (Best Model)")
    print("="*50 + "\n")
    evaluate(grid_search.best_estimator_, X_test, y_test)


if __name__ == "__main__":
    main()
