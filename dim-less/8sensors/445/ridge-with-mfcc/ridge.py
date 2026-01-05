import sys
import time
import os
import numpy as np
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
    if Y_train.ndim == 1:
        y_train_true = Y_train
    else:
        y_train_true = np.argmax(Y_train, axis=1)

    if Y_eval.ndim == 1:
        y_eval_true = Y_eval
    else:
        y_eval_true = np.argmax(Y_eval, axis=1)

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



if __name__ == "__main__":

    # parameters
    # lam = 1e4
    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])

    # state matrices
    X_train   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0-mfcc.npz")['arr_0']
    X_val     = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0-mfcc.npz")['arr_0']
    X_test    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-0.16-u_dc-0.1-mu-1.0-mfcc.npz")['arr_0']

    # Standardize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # labels
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    for lam in lambda_values:

        # Train on train set, test on test set
        results_test, cm_test = ridge_regression_fast(
            X_train, labels_train, X_test, labels_test, lam
        )

        # Train on train set, eval on val set
        results_val, cm_val = ridge_regression_fast(
            X_train, labels_train, X_val, labels_val, lam
        )

        # Save
        np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/ridge-with-mfcc/results/results_test-lam-{lam:.0e}.txt",
                results_test.reshape(1, -1), fmt="%.5f")
        np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/ridge-with-mfcc/results/conf_matrix_test-lam-{lam:.0e}.txt",
                cm_test, fmt="%.5f")

        np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/ridge-with-mfcc/results/results_val-lam-{lam:.0e}.txt",
                results_val.reshape(1, -1), fmt="%.5f")
        np.savetxt(f"/scratch/almo2783/scratch/dim-less/8sensors/445/ridge-with-mfcc/results/conf_matrix_val-lam-{lam:.0e}.txt",
                cm_val, fmt="%.5f")