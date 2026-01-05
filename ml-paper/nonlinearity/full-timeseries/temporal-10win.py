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


def process_lambda_value(
    X_train, X_val, y_train, y_val,
    lam, a, u_dc, outdir
):

    results, conf = ridge_regression_fast(
        X_train, y_train,
        X_val,   y_val,
        lam, a, u_dc
    )

    window_size = 10

    fname = (
        f"result-a-{a}-u_{u_dc}"
        f"-lam-{lam:.1e}"
        f"-win-{window_size}.npz"
    )

    np.savez(
        os.path.join(outdir, fname),
        metrics=results,
        conf_matrix=conf
    )


def fft_magnitudes_per_window(signal, n_windows=10):
    """
    Compute FFT magnitudes for each of n_windows segments.
    
    Args:
        signal: (T,) timeseries
        n_windows: int
    
    Returns:
        ffts: (n_windows, n_freqs) magnitudes
    """
    T = signal.shape[0]
    win_size = T // n_windows
    
    ffts = []
    for w in range(n_windows):
        s = w * win_size
        e = s + win_size
        seg = signal[s:e]
        fft_mag = np.abs(np.fft.rfft(seg))
        ffts.append(fft_mag)
    
    return np.stack(ffts, axis=0)


def compute_global_top_frequencies(train_matrix, n_windows=10, top_k=800, n_jobs=8):
    """
    Compute top K frequencies by average std over windows, across all train samples.
    
    Args:
        train_matrix: (n_samples, T)
        n_windows: int
        top_k: int
        n_jobs: int
    
    Returns:
        top_freq_idx: (top_k,) global indices of top varying freq bins
    """
    def single_sample_stds(signal):
        ffts = fft_magnitudes_per_window(signal, n_windows)
        return np.std(ffts, axis=0)  # (n_freqs,)
    
    # Parallel compute stds per sample
    outputs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(single_sample_stds)(train_matrix[i]) for i in range(train_matrix.shape[0])
    )
    
    # Average std across samples: (n_freqs,)
    global_std = np.mean(outputs, axis=0)
    
    # Top K indices by descending std
    top_freq_idx = np.argsort(global_std)[::-1][:top_k]
    
    return top_freq_idx


def windowed_fft_features(matrix, top_freq_idx, n_windows=10):
    """
    Extract FFT magnitudes for top frequencies across windows, for all samples.
    
    Args:
        matrix: (n_samples, T)
        top_freq_idx: (top_k,) indices
        n_windows: int
    
    Returns:
        features: (n_samples, n_windows * top_k)
    """
    n_samples, T = matrix.shape
    top_k = len(top_freq_idx)
    win_size = T // n_windows
    
    def extract_for_sample(signal):
        ffts = fft_magnitudes_per_window(signal, n_windows)  # (n_windows, n_freqs)
        # Extract top freqs: (n_windows, top_k)
        selected = ffts[:, top_freq_idx]
        # Flatten: (n_windows * top_k,)
        return selected.flatten()
    
    # Parallel over samples
    features_list = Parallel(n_jobs=8, verbose=5)(
        delayed(extract_for_sample)(matrix[i]) for i in range(n_samples)
    )
    
    return np.stack(features_list, axis=0)  # (n_samples, n_windows * top_k)


# --- Main execution ---
if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    mu = 1.0
    lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]

    train_matrix = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    val_matrix   = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")["arr_0"]

    # Load labels (assuming one-hot; code handles argmax)
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/results"
    os.makedirs(results_dir, exist_ok=True)

    n_jobs = min(8, os.cpu_count() // 2)
    N_WINDOWS = 10
    TOP_K = 800

    # --- Feature extraction ---
    print("Computing global top frequencies...")
    top_freq_idx = compute_global_top_frequencies(
        train_matrix, n_windows=N_WINDOWS, top_k=TOP_K, n_jobs=n_jobs
    )
    print(f"Top {TOP_K} freq indices computed. Shape of features will be (n_samples, {N_WINDOWS * TOP_K})")

    print("Extracting features for train...")
    X_train = windowed_fft_features(train_matrix, top_freq_idx, n_windows=N_WINDOWS)

    print("Extracting features for val...")
    X_val = windowed_fft_features(val_matrix, top_freq_idx, n_windows=N_WINDOWS)

    # --- Standardize ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --- Parallel λ sweep ---
    Parallel(n_jobs=n_jobs, verbose=1, backend='threading')(
        delayed(process_lambda_value)(
            X_train, X_val,
            labels_train, labels_val,
            lam,
            a_value, u_dc_value,
            results_dir
        )
        for lam in lambda_values
    )
    
    print("Sweep complete. Results saved to", results_dir)