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
import pandas as pd

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
    
    # First pass: energies
    for node in nodes:
        c = node.data
        e = np.sum(c**2)
        energies.append(e)
        total_energy += e
    
    energies = np.array(energies) + 1e-16
    
    # Second pass: rich features
    for i, node in enumerate(nodes):
        c = node.data
        abs_c = np.abs(c)
        
        log_energy = np.log10(energies[i])
        rel_energy = energies[i] / total_energy
        rms = np.sqrt(np.mean(c**2))
        
        mean_abs = np.mean(abs_c)
        std = np.std(c)
        skewness = skew(c)
        kurt = kurtosis(c)
        
        p = abs_c / (np.sum(abs_c) + 1e-16)
        shannon_entropy = -np.sum(p * np.log2(p + 1e-16))
        l1_l2 = np.sum(abs_c) / (np.sqrt(np.sum(c**2)) + 1e-16)
        
        zcr = np.mean(np.diff(np.sign(c)) != 0)
        crest = np.max(abs_c) / (rms + 1e-16)
        
        features.extend([
            log_energy, rel_energy, rms, mean_abs, std,
            skewness, kurt, shannon_entropy, l1_l2, zcr, crest
        ])
    
    return np.array(features, dtype=np.float32)

def build_state_matrices(train_file_list_path, val_file_list_path, test_file_list_path):
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)
    
    all_filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    
    results = Parallel(n_jobs=64, verbose=1, backend='multiprocessing')(
        delayed(process_file)(fname) for fname in all_filenames
    )
    
    state_all = np.vstack(results)
    return state_all

if __name__ == "__main__":
    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'
    
    # Load labels
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    
    labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)
    
    # Build feature matrix
    state_matrix = build_state_matrices(train_files, val_files, test_files)
    
    # K-Fold Configuration
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Results dir
    results_dir = "/scratch/almo2783/scratch/ml-paper/soft-labels/audio/lda"
    os.makedirs(results_dir, exist_ok=True)
    
    # -----------------------------
    # Bootstrapping hyperparameters
    # -----------------------------
    bootstrap_iters = 5          # Number of bootstrapping iterations (3–10 is typical)
    beta = 0.95                  # Weight on original (noisy) labels; higher = more conservative
    all_results = []
    
    # -----------------------------
    # K-Fold Training Loop
    # -----------------------------
    for fold, (train_idx, test_idx) in enumerate(kf.split(state_matrix), start=1):
        print(f"\n========== Fold {fold}/{n_splits} ==========")
        
        X_train = state_matrix[train_idx]
        X_test = state_matrix[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Ensure y_train/y_test are one-hot (required for bootstrapping)
        if y_train.ndim == 1:
            n_classes = len(np.unique(y_train))
            y_train = np.eye(n_classes)[y_train]
            y_test = np.eye(n_classes)[y_test]
        elif y_train.shape[1] == len(np.unique(np.argmax(y_train, axis=1))):
            pass  # already one-hot
        else:
            raise ValueError("Unexpected label format")
        
        # Scale per fold
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # Bootstrapping to refine (soften) training labels
        current_soft_labels = y_train.astype(np.float64).copy()
        
        for it in range(bootstrap_iters):
            current_hard_labels = np.argmax(current_soft_labels, axis=1)
            
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X_train_std, current_hard_labels)
            
            train_proba = lda.predict_proba(X_train_std)
            current_soft_labels = beta * y_train + (1 - beta) * train_proba
        
        # Final model on refined hard labels
        final_hard_labels = np.argmax(current_soft_labels, axis=1)
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train_std, final_hard_labels)
        
        # Predictions and metrics (using original labels for evaluation)
        y_train_true = np.argmax(y_train, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        
        train_pred = lda.predict(X_train_std)
        train_accuracy = accuracy_score(y_train_true, train_pred)
        
        test_pred = lda.predict(X_test_std)
        accuracy = accuracy_score(y_test_true, test_pred)
        precision = precision_score(y_test_true, test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_true, test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_true, test_pred, average='macro', zero_division=0)
        
        results = np.array([train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
        all_results.append(results)
        
        # Save
        np.savetxt(
            f"{results_dir}/fold_results-fold-{fold}.txt",
            results.reshape(1, -1),
            fmt="%.6f"
        )
    
    print("\n✅ K-Fold cross-validation with bootstrapping finished successfully.")

    all_results = np.array(all_results)
    # Save raw metrics
    np.savetxt(
        f"{results_dir}/results.txt",
        all_results.reshape(1,-1),
        fmt="%.6f"
    )
    print("\n✅ K-Fold cross-validation finished successfully.")
    pd.DataFrame(all_results, columns=["train_acc", "val_acc", "precision", "recall", "f1"]).describe()
