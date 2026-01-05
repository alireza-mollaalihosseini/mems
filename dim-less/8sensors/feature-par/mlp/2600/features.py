import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier  # Fallback for permutation


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)


def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
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

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix


def compute_feature_ranking_mlp_permutation(X, y, hidden_layer_sizes=(512, 256), max_iter=20, batch_size=128, random_state=42):
    """
    Compute feature ranking using MLP with permutation importance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features).
    y : np.ndarray
        One-hot encoded or label vector.
    hidden_layer_sizes : tuple
        Sizes of hidden layers.
    max_iter : int
        Maximum training iterations.
    batch_size : int
        Batch size for training.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ranked_idx : np.ndarray
        Indices of features sorted by importance (descending).
    importances : np.ndarray
        Mean permutation importance scores.
    """
    # Convert to labels if one-hot
    y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

    # Use sklearn MLP for simplicity with permutation (torch alternative possible but complex)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        batch_size=batch_size,
        early_stopping=True  # Optimization: prevent overfitting
    )
    mlp.fit(X, y_labels)

    # Permutation importance (optimization: n_repeats=5 for stability, but tunable)
    perm_imp = permutation_importance(mlp, X, y_labels, n_repeats=5, random_state=random_state, n_jobs=-1)
    importances = perm_imp.importances_mean
    ranked_idx = np.argsort(importances)[::-1]

    return ranked_idx, importances



def process_top_k(top_k, ranked_idx, X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir):
    selected_idx = ranked_idx[:top_k]
    X_train_selected = X_train[:, selected_idx]
    X_val_selected   = X_val[:, selected_idx]

    results, conf_matrix = ridge_regression_fast(
        X_train_selected, labels_train, X_val_selected, labels_val, 
        lambda_value, a_value, u_dc_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt",
               results.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt",
               conf_matrix, fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    lambda_value = 1e4
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
    mu = 1.0

    # Load training, validation
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/mlp/2600/results"
    os.makedirs(results_dir, exist_ok=True)

    # Compute RF scores once
    ranked_idx, rf_scores = compute_feature_ranking_mlp_permutation(X_train, labels_train)

    # Run in parallel
    Parallel(n_jobs=16)(
        delayed(process_top_k)(
            top_k, ranked_idx, X_train, X_val, labels_train, labels_val,
            lambda_value, a_value, u_dc_value, results_dir
        )
        for top_k in top_k_values
    )