import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from sklearn.model_selection import KFold
import pywt
from scipy.stats import skew, kurtosis


def lda_classification_fast(X_train, Y_train, X_eval, Y_eval):
    # -------------------------
    # Handle labels (1D or one-hot)
    # -------------------------
    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    # -------------------------
    # Train LDA
    # -------------------------
    lda = LinearDiscriminantAnalysis(
        solver="svd"   # best default for high-dimensional features
    )
    # lda = LinearDiscriminantAnalysis(
    #     solver="lsqr", shrinkage="auto"   # best default for high-dimensional features
    # ) # worser than svd

    lda.fit(X_train, y_train_true)

    # -------------------------
    # Predictions
    # -------------------------
    y_train_pred = lda.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train_true)

    y_eval_pred = lda.predict(X_eval)

    accuracy  = accuracy_score(y_eval_true, y_eval_pred)
    precision = precision_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_pred, average='macro', zero_division=0)

    results = np.array(
        [train_accuracy, accuracy, precision, recall, f1],
        dtype=np.float64
    )

    return results


def process_file(fname):
    data, _ = sf.read(fname)
    fft_vals = np.fft.rfft(data)
    # return np.abs(fft_vals).astype(np.float32)
    return np.log10(np.abs(fft_vals)+1e-16).astype(np.float32)



def build_state_matrices(train_file_list_path, val_file_list_path, test_file_list_path):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    all_filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    
    # Single parallel call to process all files (reduces joblib overhead)
    results = Parallel(n_jobs=64, verbose=1, backend='multiprocessing')(
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

    # -----------------------------
    # K-Fold Configuration
    # -----------------------------
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/lda/audio/results"
    os.makedirs(results_dir, exist_ok=True)

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
        fold_outputs = lda_classification_fast(X_train_std, y_train, X_test_std, y_test)

        # Save raw metrics
        np.savetxt(
            f"{results_dir}/fold_results-fold-{fold}-fft.txt",
            fold_outputs.reshape(1,-1),
            fmt="%.6f"
        )

    print("\n✅ K-Fold cross-validation finished successfully.")