import sys
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
# from joblib import Parallel, delayed


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
    
    return results, conf_matrix, W


if __name__ == "__main__":

    # parameters
    lam = 1e4
    extracted_features_idx = np.load('/scratch/almo2783/scratch/dim-less/8sensors/feature-par/overall_feature_idx_1057.npy')

    # state matrices
    X_train1  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val1    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test1   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train1 = X_train1[:, extracted_features_idx]
    X_test1  = X_test1[:, extracted_features_idx]
    X_val1   = X_val1[:, extracted_features_idx]

    # X_train2  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_train-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    # X_val2    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_val-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    # X_test2   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_test-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']

    # X_train3  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_train-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    # X_val3    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_val-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    # X_test3   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_test-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']

    # X_train4  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_train-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    # X_val4    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_val-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    # X_test4   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_test-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']

    # X_train5  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_train-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    # X_val5    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_val-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    # X_test5   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_test-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']

    # X_train6  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_train-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    # X_val6    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_val-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    # X_test6   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_test-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train7  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_val7    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_test7   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']

    X_train7 = X_train7[:, extracted_features_idx]
    X_test7  = X_test7[:, extracted_features_idx]
    X_val7   = X_val7[:, extracted_features_idx]

    # concatenate the matrices
    X_train = np.concatenate([X_train1, X_train7], axis=1)
    X_val   = np.concatenate([X_val1, X_val7], axis=1)
    X_test  = np.concatenate([X_test1, X_test7], axis=1)

    # Standardize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # labels
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Train on train set, test on test set
    results_test, cm_test, weights = ridge_regression_fast(
        X_train, labels_train, X_test, labels_test, lam
    )

    np.savez_compressed(f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/weights/weights-lambda-{lam}-concat-2.npx", weights)