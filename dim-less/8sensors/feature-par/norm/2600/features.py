import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


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


def top_features(W, top_k=10):
    # Exclude bias (last row of W)
    W_no_bias = W[:-1, :]

    # Compute L2 norm importance
    importance = np.linalg.norm(W_no_bias, axis=1)  # shape (n_features,)

    # Get top-k indices
    top_idx = np.argsort(importance)[::-1][:top_k]

    return top_idx


def process_top_k(top_k, X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir, weights):

    extracted_features_idx = top_features(weights, top_k=top_k)
    print(f'top_k={top_k}, Length of unique indices = {len(extracted_features_idx)}')
    X_train_selected = X_train[:, extracted_features_idx]
    X_val_selected   = X_val[:, extracted_features_idx]

    results, conf_matrix = ridge_regression_fast(
        X_train_selected, labels_train, X_val_selected, labels_val, lambda_value, a_value, u_dc_value
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
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/norm/2600/results"
    os.makedirs(results_dir, exist_ok=True)

    # Load the weights
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-{a_value}-lambda-{lambda_value}.npz")['arr_0']

    # Run in parallel
    Parallel(n_jobs=-1)(  # adjust n_jobs depending on your cluster
        delayed(process_top_k)(
            top_k, X_train, X_val, labels_train, labels_val,
            lambda_value, a_value, u_dc_value, results_dir, weights
        )
        for idx, top_k in enumerate(top_k_values)
    )
