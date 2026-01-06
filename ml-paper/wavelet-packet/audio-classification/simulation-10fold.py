import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from sklearn.model_selection import KFold
import pywt
from scipy.stats import skew, kurtosis


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


def process_file(fname, wavelet="db4", maxlevel=5, mode="symmetric"):
    data, _ = sf.read(fname)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data -= np.mean(data)
    data /= (np.max(np.abs(data)) + 1e-12)

    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order="freq")

    features = []
    total_energy = 0.0
    energies = []
    
    # -------------------------
    # First pass: energies
    # -------------------------
    for node in nodes:
        c = node.data
        e = np.sum(c**2)
        energies.append(e)
        total_energy += e

    energies = np.array(energies) + 1e-16

    # -------------------------
    # Second pass: rich features
    # -------------------------
    for i, node in enumerate(nodes):
        c = node.data
        abs_c = np.abs(c)

        # Energy features
        log_energy = np.log10(energies[i])
        rel_energy = energies[i] / total_energy
        rms = np.sqrt(np.mean(c**2))

        # Distribution
        mean_abs = np.mean(abs_c)
        std = np.std(c)
        skewness = skew(c)
        kurt = kurtosis(c)

        # Sparsity / entropy
        p = abs_c / (np.sum(abs_c) + 1e-16)
        shannon_entropy = -np.sum(p * np.log2(p + 1e-16))
        l1_l2 = np.sum(abs_c) / (np.sqrt(np.sum(c**2)) + 1e-16)

        # Temporal structure
        zcr = np.mean(np.diff(np.sign(c)) != 0)
        crest = np.max(abs_c) / (rms + 1e-16)

        features.extend([
            log_energy,
            rel_energy,
            rms,
            mean_abs,
            std,
            skewness,
            kurt,
            shannon_entropy,
            l1_l2,
            zcr,
            crest
        ])

    return np.array(features, dtype=np.float32)



def build_state_matrices(train_file_list_path, val_file_list_path, test_file_list_path):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    all_filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    
    # Single parallel call to process all files (reduces joblib overhead)
    results = Parallel(n_jobs=64, verbose=1, backend='threading')(
        delayed(process_file)(fname)
        for fname in all_filenames
    )
    
    # Stack and split
    state_all = np.vstack(results)
    
    return state_all
    


if __name__ == "__main__":

    # build matrices
    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    # Load labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    # concatenate the labels
    labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)

    # state_train, state_test = build_state_matrices(train_files, test_files)
    state_matrix = build_state_matrices(train_files, val_files, test_files)

    lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6]

    # -----------------------------
    # K-Fold Configuration
    # -----------------------------
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/audio-classification/results"
    os.makedirs(results_dir, exist_ok=True)

    n_lams = len(lambda_values)
    
    # container: results[lambda_idx][fold] -> 6 metrics
    results_per_lambda = [ [] for _ in range(n_lams) ]

    # -----------------------------
    # K-Fold Training Loop
    # -----------------------------
    for fold, (train_idx, test_idx) in enumerate(
            kf.split(state_matrix), start=1):

        print(f"\n========== Fold {fold}/{n_splits} ==========")

        # split
        X_train = state_matrix[train_idx]
        X_test  = state_matrix[test_idx]

        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        # scale PER FOLD (no leakage)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std  = scaler.transform(X_test)

        # evaluate all lambdas (parallel)
        fold_outputs = Parallel(
            n_jobs=64,
            verbose=1,
            backend="multiprocessing"
        )(
            delayed(ridge_regression_fast)(
                X_train_std, y_train,
                X_test_std,  y_test,
                lam
            )
            for lam in lambda_values
        )

        # -----------------------------
        # Save per-fold results
        # -----------------------------
        for i, (metrics) in enumerate(fold_outputs):

            lam = lambda_values[i]

            # store for mean/std later
            results_per_lambda[i].append(metrics)

            # Save raw metrics
            np.savetxt(
                f"{results_dir}/fold_results-lambda-{lam:.1e}-fold-{fold}.txt",
                metrics.reshape(1,-1),
                fmt="%.6f"
            )

    # -----------------------------
    # Compute Mean ± Std over folds
    # -----------------------------
    for i, lam in enumerate(lambda_values):

        fold_data = np.vstack(results_per_lambda[i])   # shape: (n_folds,6)

        mean = fold_data.mean(axis=0)
        std  = fold_data.std(axis=0)

        summary = np.vstack([
            mean,
            std
        ])

        np.savetxt(
            f"{results_dir}/summary-lambda-{lam:.1e}.txt",
            summary,
            header="ROW1=MEAN  ROW2=STD\n"
                "COLS=[lambda, train_acc, test_acc, precision, recall, f1]",
            fmt="%.8f"
        )

    print("\n✅ K-Fold cross-validation finished successfully.")