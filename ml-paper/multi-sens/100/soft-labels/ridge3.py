import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
plt.style.use('ggplot')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.special import softmax

if __name__ == '__main__':
    # You can keep your wide range — SVD handles extremes fine — or narrow it for slightly faster tuning
    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                              1e2, 1e3, 1e4, 1e5, 1e6])
    # Or a more moderate range (faster, still plenty):
    # lambda_values = np.logspace(-10, 10, 21)

    f_values = np.linspace(1000, 50000, 101)
    num_samples = 10910
    num_tenths = 10
    num_classes = 10  # fixed

    # Load full 1-second state matrix
    state_matrix_full = np.zeros((num_samples, len(f_values) * 60))
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix_full[:, i*60:(i+1)*60] = cols

    # Load 10 one-tenth state matrices
    sub_state_matrices = []
    for seg in range(1, num_tenths + 1):
        state_mat_seg = np.zeros((num_samples, len(f_values) * 60))
        for i, f in enumerate(f_values):
            cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/one-tenth/f-{int(f)}-{seg}.npz")["arr_0"]
            state_mat_seg[:, i*60:(i+1)*60] = cols
        sub_state_matrices.append(state_mat_seg)

    # Labels (one-hot)
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    hard_y_train = np.argmax(labels_train, axis=1)
    hard_y_val   = np.argmax(labels_val, axis=1)
    hard_y_test  = np.argmax(labels_test, axis=1)

    num_train = labels_train.shape[0]
    num_val   = labels_val.shape[0]

    # Splits
    train_full = state_matrix_full[:num_train]
    val_full   = state_matrix_full[num_train:num_train + num_val]
    test_full  = state_matrix_full[num_train + num_val:]

    sub_train_list = [m[:num_train] for m in sub_state_matrices]
    sub_val_list   = [m[num_train:num_train + num_val] for m in sub_state_matrices]
    sub_test_list  = [m[num_train + num_val:] for m in sub_state_matrices]

    # Teacher ensemble (10 sub-models)
    probs_train_list = []
    probs_val_list   = []
    probs_test_list  = []

    for k in range(num_tenths):
        print(f"\nTraining teacher sub-model {k+1}/{num_tenths}")

        scaler_k = StandardScaler()
        X_train_std = scaler_k.fit_transform(sub_train_list[k])
        X_val_std   = scaler_k.transform(sub_val_list[k])
        X_test_std  = scaler_k.transform(sub_test_list[k])

        # Add bias
        X_train_b = np.hstack((X_train_std, np.ones((X_train_std.shape[0], 1))))
        X_val_b   = np.hstack((X_val_std,   np.ones((X_val_std.shape[0],   1))))
        X_test_b  = np.hstack((X_test_std,  np.ones((X_test_std.shape[0],  1))))

        # SVD (economy)
        U, S, Vt = np.linalg.svd(X_train_b, full_matrices=False)

        proj_Y = U.T @ labels_train                                          # r × 10
        Z_train = X_train_b @ Vt.T                                           # n_train × r
        Z_val   = X_val_b   @ Vt.T                                           # n_val   × r
        Z_test  = X_test_b  @ Vt.T                                           # n_test  × r

        # Lambda tuning (very fast now)
        best_val_acc = -np.inf
        best_lam = None
        for lam in lambda_values:
            coeff = S / (S**2 + lam)
            filtered = coeff[:, np.newaxis] * proj_Y
            val_logits = Z_val @ filtered
            val_acc = accuracy_score(hard_y_val, np.argmax(val_logits, axis=1))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lam = lam

        print(f"  Best λ = {best_lam:.1e}, val accuracy = {best_val_acc:.4f}")

        # Final forward pass with best λ
        coeff = S / (S**2 + best_lam)
        filtered = coeff[:, np.newaxis] * proj_Y
        probs_train_list.append(softmax(Z_train @ filtered, axis=1))
        probs_val_list.append(softmax(Z_val @ filtered, axis=1))
        probs_test_list.append(softmax(Z_test @ filtered, axis=1))

    # Soft targets for distillation
    soft_Y_train = np.mean(probs_train_list, axis=0)

    # Teacher ensemble baseline
    ensemble_val_probs = np.mean(probs_val_list, axis=0)
    ensemble_test_probs = np.mean(probs_test_list, axis=0)
    ensemble_val_acc = accuracy_score(hard_y_val, np.argmax(ensemble_val_probs, axis=1))
    ensemble_test_acc = accuracy_score(hard_y_test, np.argmax(ensemble_test_probs, axis=1))
    print(f"\nTeacher ensemble val accuracy: {ensemble_val_acc:.4f}")
    print(f"Teacher ensemble test accuracy: {ensemble_test_acc:.4f}")

    # Distilled student on full 1-second features
    print("\nTraining distilled student model on full 1-second features")
    scaler_full = StandardScaler()
    X_train_std = scaler_full.fit_transform(train_full)
    X_val_std   = scaler_full.transform(val_full)
    X_test_std  = scaler_full.transform(test_full)

    X_train_b = np.hstack((X_train_std, np.ones((X_train_std.shape[0], 1))))
    X_val_b   = np.hstack((X_val_std,   np.ones((X_val_std.shape[0],   1))))
    X_test_b  = np.hstack((X_test_std,  np.ones((X_test_std.shape[0],  1))))

    U, S, Vt = np.linalg.svd(X_train_b, full_matrices=False)

    proj_soft = U.T @ soft_Y_train
    Z_train = X_train_b @ Vt.T
    Z_val   = X_val_b   @ Vt.T
    Z_test  = X_test_b  @ Vt.T

    best_val_acc = -np.inf
    best_lam = None
    for lam in lambda_values:
        coeff = S / (S**2 + lam)
        filtered = coeff[:, np.newaxis] * proj_soft
        val_logits = Z_val @ filtered
        val_acc = accuracy_score(hard_y_val, np.argmax(val_logits, axis=1))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lam = lam

    print(f"Distilled best λ = {best_lam:.1e}, val accuracy = {best_val_acc:.4f}")

    # Final test performance
    coeff = S / (S**2 + best_lam)
    filtered = coeff[:, np.newaxis] * proj_soft
    test_logits = Z_test @ filtered
    test_hats = np.argmax(test_logits, axis=1)

    test_acc = accuracy_score(hard_y_test, test_hats)
    test_prec = precision_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_rec = recall_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_f1 = f1_score(hard_y_test, test_hats, average='macro', zero_division=0)

    print(f"\nDistilled student test accuracy:  {test_acc:.4f}")
    print(f"Distilled student test precision: {test_prec:.4f}")
    print(f"Distilled student test recall:    {test_rec:.4f}")
    print(f"Distilled student test F1:        {test_f1:.4f}")