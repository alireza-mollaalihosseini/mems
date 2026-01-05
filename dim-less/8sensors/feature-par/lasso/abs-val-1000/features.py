import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV


def top_features(W, top_k=10):
    # Exclude bias (last row of W)
    W_no_bias = W[:-1, :]

    # Dict to store max importance per feature across all classes
    feature_importance = {}

    # Loop over each class (columns of W)
    for c in range(W_no_bias.shape[1]):
        importance = np.abs(W_no_bias[:, c])
        top_idx = np.argsort(importance)[::-1][:top_k]

        # Save the *maximum importance* seen across classes
        for idx in top_idx:
            if idx not in feature_importance:
                feature_importance[idx] = importance[idx]
            else:
                feature_importance[idx] = max(feature_importance[idx], importance[idx])

    # Sort collected unique indices by importance (descending)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Pick exactly top_k
    top_idx_final = np.array([idx for idx, _ in sorted_features[:top_k]])
    
    return top_idx_final


# def lasso_regression_fast(X_train, Y_train, X_eval, Y_eval, a, u_dc, results_dir, lam):
#     # Add bias term
#     X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
#     X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

#     # Train Lasso regression (multi-output)
#     lasso = MultiOutputRegressor(Lasso(alpha=lam, max_iter=10000, tol=1e-4))
#     lasso.fit(X_train_b, Y_train)

#     # Predictions
#     y_train_pred = lasso.predict(X_train_b)
#     y_eval_pred  = lasso.predict(X_eval_b)

#     # Convert one-hot encoded labels to class indices
#     y_train_true = np.argmax(Y_train, axis=1)
#     y_eval_true  = np.argmax(Y_eval,  axis=1)

#     # Predicted classes
#     y_train_hats = np.argmax(y_train_pred, axis=1)
#     y_eval_hats  = np.argmax(y_eval_pred,  axis=1)

#     # Compute metrics
#     train_accuracy = np.mean(y_train_hats == y_train_true)
#     accuracy  = accuracy_score(y_eval_true, y_eval_hats)
#     precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
#     f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

#     conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)

#     results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
#     np.savetxt(
#         f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-lasso.txt",
#         results.reshape(1, -1), fmt="%.6f"
#     )

#     np.savetxt(
#         f"{results_dir}/conf_matrix-a-{a:.2f}-u_dc-{u_dc:.2f}-lasso.txt",
#         conf_matrix, fmt="%.5f"
#     )

#     return results, conf_matrix


# def process_grid_search(X_train, X_val, labels_train, labels_val, a_value, u_dc_value, results_dir, lam):
#     results, conf_matrix = lasso_regression_fast(
#         X_train, labels_train, X_val, labels_val, a_value, u_dc_value, results_dir, lam
#     )
#     return results

def lasso_gridsearch_fast(X_train, Y_train, X_eval, Y_eval, a, u_dc, results_dir):
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Define base model
    base_lasso = Lasso(random_state=42)

    # Wrap in MultiOutputRegressor
    multi_lasso = MultiOutputRegressor(base_lasso, n_jobs=-1)

    # Define parameter grid
    param_grid = {
        "estimator__alpha": np.logspace(-4, 3, 8),   # 1e-6 → 1e5
        "estimator__max_iter": [20000],
        "estimator__tol": [1e-5]
    }

    # GridSearchCV setup
    grid = GridSearchCV(
        estimator=multi_lasso,
        param_grid=param_grid,
        scoring="accuracy",     # can be replaced with "neg_mean_squared_error"
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    # Fit model
    grid.fit(X_train_b, Y_train)

    # Best estimator
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    # Predictions
    y_train_pred = best_model.predict(X_train_b)
    y_eval_pred  = best_model.predict(X_eval_b)

    # Convert one-hot to class indices
    y_train_true = np.argmax(Y_train, axis=1)
    y_eval_true  = np.argmax(Y_eval, axis=1)

    y_train_hats = np.argmax(y_train_pred, axis=1)
    y_eval_hats  = np.argmax(y_eval_pred, axis=1)

    # Compute metrics
    train_accuracy = np.mean(y_train_hats == y_train_true)
    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)

    # Save results
    results = np.array([a, u_dc, train_accuracy, accuracy, precision, recall, f1, best_score], dtype=np.float64)
    np.savetxt(
        f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-lasso-grid.txt",
        results.reshape(1, -1), fmt="%.6f"
    )

    np.savetxt(
        f"{results_dir}/conf_matrix-a-{a:.2f}-u_dc-{u_dc:.2f}-lasso-grid.txt",
        conf_matrix, fmt="%.5f"
    )

    # Save best hyperparameters
    with open(f"{results_dir}/best_params-a-{a:.2f}-u_dc-{u_dc:.2f}-lasso-grid.txt", "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    return results, conf_matrix


def process_grid_search(X_train, X_val, labels_train, labels_val, a_value, u_dc_value, results_dir):
    results, conf_matrix = lasso_gridsearch_fast(
        X_train, labels_train, X_val, labels_val, a_value, u_dc_value, results_dir
    )
    return results


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    mu = 1.0
    lam = 0.01

    # Load training and validation data
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Load the weights
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']
    extracted_features_idx = top_features(weights, top_k=1000)
    X_train = X_train[:, extracted_features_idx]
    X_val   = X_val[:, extracted_features_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Results directory
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/lasso/abs-val-1000/results"
    os.makedirs(results_dir, exist_ok=True)

    # # Run grid search (single combination here, but can be extended)
    # process_grid_search(X_train, X_val, labels_train, labels_val, a_value, u_dc_value, results_dir, lam)

    # Run grid search (single combination here, but can be extended)
    process_grid_search(X_train, X_val, labels_train, labels_val, a_value, u_dc_value, results_dir)
