import shap
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    A = X_train.T @ X_train + lam * I
    b = X_train.T @ Y_train
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam):
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Train ridge regression
    W = ridge_closed_form(X_train_b, Y_train, lam)

    # Handle 1D vs 2D labels
    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    # Predictions
    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)

    return results, W


state_matrix = np.load("/scratch/almo2783/scratch/test/state-matrix-64-sens.npz")["arr_0"]

# labels
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
lebels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

train_state = state_matrix[:len(labels_train)]
val_state   = state_matrix[len(labels_train): len(labels_train) + len(labels_val)]
test_state  = state_matrix[len(labels_train) + len(labels_val): ]

# scale PER FOLD (no leakage)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(train_state)
X_val_std   = scaler.transform(val_state)
X_test_std  = scaler.transform(test_state)


best_lambda = 10.0
# ridge regression best model on test set
best_results, W_best = ridge_regression_fast(
    X_train_std, labels_train,
    X_val_std, labels_val,
    best_lambda
)


def ridge_predict(X):
    # Add bias term
    X_b = np.hstack((X, np.ones((X.shape[0], 1))))
    return X_b @ W_best

X_background = shap.utils.sample(X_train_std, 100)
masker = shap.maskers.Independent(X_background)

shap_values_per_class = {}

for c in range(W_best.shape[1]):
    def predict_c(X, c=c):
        return ridge_predict(X)[:, c]

    explainer = shap.Explainer(predict_c, masker, max_evals= 10000)
    shap_values_per_class[c] = explainer(X_val_std)



n_classes = W_best.shape[1]
n_features = X_test_std.shape[1]

mean_shap_per_class = np.zeros((n_classes, n_features))

for c in range(n_classes):
    # SHAP matrix E^{(c)} ∈ R^{n × m}
    E_c = shap_values_per_class[c].values
    
    # Mean absolute SHAP per feature
    mean_shap_per_class[c] = np.mean(np.abs(E_c), axis=0)


global_shap_importance = np.sum(mean_shap_per_class, axis=0)


feature_ranking = np.argsort(global_shap_importance)[::-1]

ranked_features = [
    (idx, global_shap_importance[idx])
    for idx in feature_ranking
]

np.save("shap_feature_ranking_64_sens.npy", feature_ranking)
np.save("ridge_shap_feature_ranking_64_sens.npy", np.array(ranked_features))