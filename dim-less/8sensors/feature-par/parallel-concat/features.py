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

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix


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


def process_top_k(top_k, lambda_value, results_dir):

    extracted_features_idx1 = top_features(weights1, top_k=top_k)
    extracted_features_idx7 = top_features(weights7, top_k=top_k)
    
    X_train1_selected = X_train1_full[:, extracted_features_idx1]
    X_test1_selected  = X_test1_full[:, extracted_features_idx1]
    X_val1_selected   = X_val1_full[:, extracted_features_idx1]

    X_train7_selected = X_train7_full[:, extracted_features_idx7]
    X_test7_selected  = X_test7_full[:, extracted_features_idx7]
    X_val7_selected   = X_val7_full[:, extracted_features_idx7]

    # Concatenate
    X_train = np.concatenate([X_train1_selected, X_train7_selected], axis=1)
    X_test  = np.concatenate([X_test1_selected, X_test7_selected], axis=1)
    X_val   = np.concatenate([X_val1_selected, X_val7_selected], axis=1)

    # Standardize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_val   = scaler.transform(X_val)

    # Ridge regression
    results_test, cm_test = ridge_regression_fast(
        X_train, labels_train, X_test, labels_test, lambda_value
    )
    results_val, cm_val = ridge_regression_fast(
        X_train, labels_train, X_val, labels_val, lambda_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results_test-topk-{top_k}-lambda-{lambda_value}.txt",
               results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_test-topk-{top_k}-lambda-{lambda_value}.txt",
               cm_test, fmt="%.5f")

    np.savetxt(f"{results_dir}/results_val-topk-{top_k}-lambda-{lambda_value}.txt",
               results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_val-topk-{top_k}-lambda-{lambda_value}.txt",
               cm_val, fmt="%.5f")


if __name__ == "__main__":
    
    lambda_value = 1e4
    # top_k = 500
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
    mu = 1.0

    # --- Load full matrices ONCE ---
    X_train1_full = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val1_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test1_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train7_full = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_val7_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_test7_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results"
    os.makedirs(results_dir, exist_ok=True)

    # Load the weights
    weights1 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights-a-0.16-lambda-10000.0.npz")['arr_0']
    weights7 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']

    # Run in parallel
    Parallel(n_jobs=-1)(  # adjust n_jobs depending on your cluster
        delayed(process_top_k)(
            top_k, lambda_value, results_dir
        )
        for idx, top_k in enumerate(top_k_values)
    )

    # extracted_features_idx1 = top_features(weights1, top_k=top_k)
    # extracted_features_idx7 = top_features(weights7, top_k=top_k)
    
    # X_train1_selected = X_train1_full[:, extracted_features_idx1]
    # X_test1_selected  = X_test1_full[:, extracted_features_idx1]
    # X_val1_selected   = X_val1_full[:, extracted_features_idx1]

    # X_train7_selected = X_train7_full[:, extracted_features_idx7]
    # X_test7_selected  = X_test7_full[:, extracted_features_idx7]
    # X_val7_selected   = X_val7_full[:, extracted_features_idx7]

    # # Concatenate
    # X_train = np.concatenate([X_train1_selected, X_train7_selected], axis=1)
    # X_test  = np.concatenate([X_test1_selected, X_test7_selected], axis=1)
    # X_val   = np.concatenate([X_val1_selected, X_val7_selected], axis=1)

    # # Standardize
    # scaler  = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test  = scaler.transform(X_test)
    # X_val   = scaler.transform(X_val)

    # # Ridge regression
    # results_test, cm_test = ridge_regression_fast(
    #     X_train, labels_train, X_test, labels_test, lambda_value
    # )
    # results_val, cm_val = ridge_regression_fast(
    #     X_train, labels_train, X_val, labels_val, lambda_value
    # )

    # # Save results
    # np.savetxt(f"{results_dir}/results_test-topk-{top_k}-lambda-{lambda_value}.txt",
    #            results_test.reshape(1, -1), fmt="%.5f")
    # np.savetxt(f"{results_dir}/conf_matrix_test-topk-{top_k}-lambda-{lambda_value}.txt",
    #            cm_test, fmt="%.5f")

    # np.savetxt(f"{results_dir}/results_val-topk-{top_k}-lambda-{lambda_value}.txt",
    #            results_val.reshape(1, -1), fmt="%.5f")
    # np.savetxt(f"{results_dir}/conf_matrix_val-topk-{top_k}-lambda-{lambda_value}.txt",
    #            cm_val, fmt="%.5f")
