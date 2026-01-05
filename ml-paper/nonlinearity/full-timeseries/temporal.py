import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)


def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
    # # Add bias term
    # X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    # X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Train ridge regression
    # W = ridge_closed_form(X_train_b, Y_train, lam)
    W = ridge_closed_form(X_train, Y_train, lam)

    # Handle 1D vs 2D labels
    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    # Predictions
    # y_train_pred = X_train_b @ W
    y_train_pred = X_train @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    # y_eval_pred = X_eval_b @ W
    y_eval_pred = X_eval @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix


def sliding_fft_features(X, window_size, stride):
    """
    X: (n_samples, n_timeseries)   raw state matrix
    window_size: FFT size
    stride: window hop size

    Returns:
    (n_samples, n_windows*(window_size//2+1))
    """

    n_samples, T = X.shape
    n_windows = (T - window_size) // stride + 1
    n_freq = window_size // 2 + 1

    features = np.zeros((n_samples, n_windows * n_freq), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end   = start + window_size
        seg = X[:, start:end]

        # FFT magnitude
        fft_vals = np.abs(np.fft.rfft(seg, axis=1))

        # place block
        s = i * n_freq
        e = s + n_freq
        features[:, s:e] = fft_vals

    return features


def process_lambda_value(
    X_train, X_val, y_train, y_val,
    lam, a, u_dc, window_size, stride,
    outdir
):

    results, conf = ridge_regression_fast(
        X_train, y_train,
        X_val,   y_val,
        lam, a, u_dc
    )

    fname = (
        f"result-a-{a}-u_{u_dc}"
        f"-lam-{lam:.1e}"
        f"-win-{window_size}"
        f"-stride-{stride}.npz"
    )

    np.savez(
        os.path.join(outdir, fname),
        metrics=results,
        conf_matrix=conf
    )


a_value = 0.44
u_dc_value = 0.4
mu = 1.0
lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
# window_sizes = [4096, 8192, 16384, 32768]
# strides      = [2048, 4096, 8192]
window_sizes = [4096]
strides      = [32768]

train_matrix = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
val_matrix   = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")["arr_0"]
# test_matrix  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz")["arr_0"]

# Load labels
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
# labels_test   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

# Results dir
results_dir = f"/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/results"
os.makedirs(results_dir, exist_ok=True)

n_jobs = min(8, os.cpu_count() // 2)

# apply sliding windows for temporal fft correlations
for win in window_sizes:
    for stride in strides:

        print(f"\n=== FFT window={win}, stride={stride} ===")

        # --- feature extraction ---
        X_train = sliding_fft_features(train_matrix, win, stride)
        X_val   = sliding_fft_features(val_matrix,   win, stride)

        # --- standardize ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # --- parallel λ sweep ---
        Parallel(n_jobs=n_jobs, verbose=1, backend='threading')(
            delayed(process_lambda_value)(
                X_train, X_val,
                labels_train, labels_val,
                lam,
                a_value, u_dc_value,
                win, stride,
                results_dir
            )
            for lam in lambda_values
        )