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



def process_top_k(top_k, ranked_idx, X_train, X_val, labels_train, labels_val, lambda_values, a_value, u_dc_value, results_dir):
    selected_idx = ranked_idx[:top_k]
    X_train_selected = X_train[:, selected_idx]
    X_val_selected   = X_val[:, selected_idx]

    # Standardize
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_val_selected   = scaler.transform(X_val_selected)

    best_val_acc = -np.inf
    best_lambda = None
    best_results = []

    for lambda_value in lambda_values:

        results = ridge_regression_fast(
            X_train_selected, labels_train, X_val_selected, labels_val, 
            lambda_value
        )

        if results[2] > best_val_acc:
            best_val_acc   = results[2]
            best_lambda    = results[0]
            best_results = results


    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-opt.txt",
               np.array(best_results).reshape(1, -1), fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.9
    u_dc_value = 1.0
    # lambda_value = 1e2
    lambda_values = np.array([1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4])
    top_k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 2500, 3000, 3500, 3840]
    mu = 1.0

    # f_values = np.linspace(1000, 50000, 101)
    f_values = np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
       42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
       38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
        4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
       38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
       40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
       21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
       35790])
    f_values = np.sort(f_values)

    state_matrix = np.zeros((10910, len(f_values) * 60))

    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*60:(i+1)*60] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    lebels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    X_train = state_matrix[:len(labels_train)]
    X_val   = state_matrix[len(labels_train): len(labels_train) + len(labels_val)]
    X_test  = state_matrix[len(labels_train) + len(labels_val): ]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/top_64/gini/results"
    os.makedirs(results_dir, exist_ok=True)

    # load scores from gini
    ranked_idx = np.load("/scratch/almo2783/scratch/ml-paper/multi-sens/100/top_64/gini/feature_ranking_idx.npy")
    rf_scores = np.load("/scratch/almo2783/scratch/ml-paper/multi-sens/100/top_64/gini/feature_importances.npy")

    # Run in parallel
    Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(process_top_k)(
            top_k, ranked_idx, X_train, X_val, labels_train, labels_val,
            lambda_values, a_value, u_dc_value, results_dir
        )
        for top_k in top_k_values
    )
