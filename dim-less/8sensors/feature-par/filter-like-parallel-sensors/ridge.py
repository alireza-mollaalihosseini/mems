import sys
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


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


def process_lambda(lam, X_train, labels_train, X_test, labels_test, X_val, labels_val, results_dir):
    # Train on train set, test on test set
    results_test, cm_test = ridge_regression_fast(
        X_train, labels_train, X_test, labels_test, lam
    )

    # Train on train set, eval on val set
    results_val, cm_val = ridge_regression_fast(
        X_train, labels_train, X_val, labels_val, lam
    )

    # Save results
    np.savetxt(f"{results_dir}/results_test-lam-{lam:.0e}.txt",
               results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_test-lam-{lam:.0e}.txt",
               cm_test, fmt="%.5f")

    np.savetxt(f"{results_dir}/results_val-lam-{lam:.0e}.txt",
               results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_val-lam-{lam:.0e}.txt",
               cm_val, fmt="%.5f")

    return lam


if __name__ == "__main__":

    # parameters
    # lam = 1e4
    lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                              1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
    results_dir = "/scratch/almo2783/scratch/dim-less/8sensors/feature-par/filter-like-parallel-sensors/results"
    os.makedirs(results_dir, exist_ok=True)

    # state matrices
    X_train1  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val1    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test1   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train1  = X_train1[:, 0:480]
    X_val1    = X_val1[:, 0:480]
    X_test1   = X_test1[:, 0:480]

    X_train2  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_train-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_val2    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_val-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_test2   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_test-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']

    X_train2  = X_train2[:, 460:550]
    X_val2    = X_val2[:, 460:550]
    X_test2   = X_test2[:, 460:550]

    X_train3  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_train-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_val3    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_val-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_test3   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_test-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']

    X_train3  = X_train3[:, 530:600]
    X_val3    = X_val3[:, 530:600]
    X_test3   = X_test3[:, 530:600]

    X_train4  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_train-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val4    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_val-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test4   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_test-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train4  = X_train4[:, 580:900]
    X_val4    = X_val4[:, 580:900]
    X_test4   = X_test4[:, 580:900]

    X_train5  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_train-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    X_val5    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_val-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    X_test5   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_test-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']

    X_train5  = X_train5[:, 820:1200]
    X_val5    = X_val5[:, 820:1200]
    X_test5   = X_test5[:, 820:1200]

    X_train6  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_train-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val6    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_val-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test6   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_test-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train6  = X_train6[:, 1100:2000]
    X_val6    = X_val6[:, 1100:2000]
    X_test6   = X_test6[:, 1100:2000]

    X_train7  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_val7    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_test7   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']

    X_train7  = X_train7[:, 1800:3500]
    X_val7    = X_val7[:, 1800:3500]
    X_test7   = X_test7[:, 1800:3500]

    # concatenate the matrices
    X_train = np.concatenate([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7], axis=1)
    X_val   = np.concatenate([X_val1, X_val2, X_val3, X_val4, X_val5, X_val6, X_val7], axis=1)
    X_test  = np.concatenate([X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7], axis=1)

    # Standardize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # labels
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Parallel execution (n_jobs=-1 uses all available cores)
    processed = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_lambda)(
            lam, X_train, labels_train, X_test, labels_test,
            X_val, labels_val, results_dir
        )
        for lam in lambda_values
    )
