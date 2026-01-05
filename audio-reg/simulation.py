import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import soundfile as sf


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


def process_file(fname):
    data, _ = sf.read(fname)
    fft_vals = np.fft.rfft(data)
    return np.abs(fft_vals).astype(np.float32)



def build_state_matrices(train_file_list_path, val_file_list_path):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    
    # Combine for single parallel processing
    all_filenames = np.concatenate([train_filenames, val_filenames])
    n_train = len(train_filenames)
    
    # Single parallel call to process all files (reduces joblib overhead)
    results = Parallel(n_jobs=64, verbose=1, backend='threading')(
        delayed(process_file)(fname)
        for fname in all_filenames
    )
    
    # Stack and split
    state_all = np.vstack(results)
    state_train = state_all[:n_train]
    state_val = state_all[n_train:]
    
    return state_train, state_val

    


if __name__ == "__main__":

    # build matrices
    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    # val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    state_train, state_test = build_state_matrices(train_files, test_files)
    # state_train, state_val, state_test = build_state_matrices(train_files, val_files, test_files)

    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(state_train)
    # state_val_std = scaler.transform(state_val)
    state_test_std = scaler.transform(state_test)

    
    lambda_values = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18]
    
    # Load labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    # labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/audio-reg/results"
    os.makedirs(results_dir, exist_ok=True)

    # Parallelize over lambdas (each independent; use loky for CPU-bound linear algebra)
    lambda_results = Parallel(n_jobs=-1, verbose=1, backend='loky')(
        delayed(ridge_regression_fast)(state_train_std, labels_train, state_test_std, labels_test, lam)
        for lam in lambda_values
    )
    
    # # Sequential save (I/O bound, but low overhead)
    # for i, (results, conf_matrix) in enumerate(lambda_results):
    #     lambda_value = lambda_values[i]
    #     np.savetxt(f"{results_dir}/results-lambda-{lambda_value:.1e}.txt",
    #                results.reshape(1, -1), fmt="%.5f")
    #     np.savetxt(f"{results_dir}/conf_matrix-lambda-{lambda_value:.1e}.txt",
    #                conf_matrix, fmt="%.5f")

    # Sequential save (I/O bound, but low overhead)
    for i, (results, conf_matrix) in enumerate(lambda_results):
        lambda_value = lambda_values[i]
        np.savetxt(f"{results_dir}/results-test-lambda-{lambda_value:.1e}.txt",
                   results.reshape(1, -1), fmt="%.5f")
        np.savetxt(f"{results_dir}/conf_matrix-test-lambda-{lambda_value:.1e}.txt",
                   conf_matrix, fmt="%.5f")