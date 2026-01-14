import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
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
    
    return results



def process_top_k(top_k, ranked_idx, X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir):
    selected_idx = ranked_idx[:top_k]
    X_train_selected = X_train[:, selected_idx]
    X_val_selected   = X_val[:, selected_idx]

    # Standardize
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_val_selected   = scaler.transform(X_val_selected)

    results = ridge_regression_fast(
        X_train_selected, labels_train, X_val_selected, labels_val, 
        lambda_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt",
               results.reshape(1, -1), fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.9
    u_dc_value = 1.0
    lambda_value = 1e2
    # top_k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
    # top_k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 20, 25, 30, 35, 40, 45, 50, 100, 
    #             200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1500, 2000, 2500, 3000, 
    #             3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 11000, 12000, 13000, 13248]
    top_k_values = [1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
       2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500]
    mu = 1.0

    f_values = np.linspace(1000, 50000, 36)

    state_matrix = np.zeros((10510, len(f_values) * 368))

    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a_value:.2f}-u_dc-{u_dc_value:.2f}/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*368:(i+1)*368] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    X_train = state_matrix[:len(labels_train)]
    X_val   = state_matrix[len(labels_train):]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/gini/results"
    os.makedirs(results_dir, exist_ok=True)

    # load scores from gini
    ranked_idx = np.load("/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/gini/feature_ranking_idx.npy")
    rf_scores = np.load("/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/gini/feature_importances.npy")

    # Run in parallel
    Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(process_top_k)(
            top_k, ranked_idx, X_train, X_val, labels_train, labels_val,
            lambda_value, a_value, u_dc_value, results_dir
        )
        for top_k in top_k_values
    )
