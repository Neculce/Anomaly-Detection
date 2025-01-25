shuttle = loadmat('./shuttle.mat')
X = shuttle['X']
y = shuttle['y']


balanced_accuracies = []
roc_auc_scores = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=np.random.randint(0, 1000) + i)

    X_train = standardizer(X_train)
    X_test = standardizer(X_test)

    # Fit IForest, LODA and DIF using the training data and compute the
    # balanced accuracy (BA) and the area under the curve (ROC AUC-
    # using sklearn.metrics.roc auc score) for each model. Compute the
    # mean BA and ROC AUC obtained for 10 different train-test splits for
    # each of the models.

    Iforest = iforest.IForest(contamination=0.02)
    Loda = loda.LODA(contamination=0.02)
    Dif = dif.DIF(contamination=0.02)

    Iforest.fit(X_train)
    Loda.fit(X_train)
    Dif.fit(X_train)

    y_iforest = Iforest.decision_function(X_test)
    y_loda = Loda.decision_function(X_test)
    y_dif = Dif.decision_function(X_test)

    threshold = 0.3

    y_iforest = (y_iforest > np.percentile(y_iforest, 100 * (1 - threshold))).astype(int)
    y_loda = (y_loda > np.percentile(y_loda, 100 * (1 - threshold))).astype(int)
    y_dif = (y_dif > np.percentile(y_dif, 100 * (1 - threshold))).astype(int)
    
    balanced_accuracies.append([balanced_accuracy_score(y_test, y_iforest), balanced_accuracy_score(y_test, y_loda), balanced_accuracy_score(y_test, y_dif)])
    roc_auc_scores.append([roc_auc_score(y_test, y_iforest), roc_auc_score(y_test, y_loda), roc_auc_score(y_test, y_dif)])

    print(f"Split {i+1} done!")
    print(f"Balanced Accuracies: {balanced_accuracies[-1]}")
    print(f"ROC AUC Scores: {roc_auc_scores[-1]}")

balanced_accuracies = np.array(balanced_accuracies)
roc_auc_scores = np.array(roc_auc_scores)

print(f"Mean Balanced Accuracies: {balanced_accuracies.mean(axis=0)}")
print(f"Mean ROC AUC Scores: {roc_auc_scores.mean(axis=0)}")