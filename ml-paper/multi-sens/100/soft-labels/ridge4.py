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



def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    A = X_train.T @ X_train + lam * I
    b = X_train.T @ Y_train
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b

def compute_ridge_weights(X_train_std, Y, lam):
    X_train_b = np.hstack((X_train_std, np.ones((X_train_std.shape[0], 1), dtype=X_train_std.dtype)))
    return ridge_closed_form(X_train_b, Y, lam)

def predict_logits(X_std, W):
    X_b = np.hstack((X_std, np.ones((X_std.shape[0], 1), dtype=X_std.dtype)))
    return X_b @ W

def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam,
                          hard_y_train=None, hard_y_eval=None):
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_train.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)

    if hard_y_train is None:
        hard_y_train = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    if hard_y_eval is None:
        hard_y_eval = Y_eval if Y_eval.ndim == 1 else np.argmax(Y_eval, axis=1)

    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == hard_y_train)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(hard_y_eval, y_eval_hats)
    precision = precision_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(hard_y_eval, y_eval_hats, average='macro', zero_division=0)
    
    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1])
    
    return results

if __name__ == '__main__':
    lambda_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                              1e2, 1e3, 1e4, 1e5, 1e6])
    f_values = np.linspace(1000, 50000, 101)
    num_samples = 10910
    num_tenths = 10

    # Load full 1-second state matrix
    state_matrix_full = np.zeros((num_samples, len(f_values) * 60))
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix_full[:, i*60:(i+1)*60] = cols

    # Load 10 sub-segment (0.1 s) state matrices
    sub_state_matrices = []
    for seg in range(1, num_tenths + 1):
        state_mat_seg = np.zeros((num_samples, len(f_values) * 60))
        for i, f in enumerate(f_values):
            cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/one-tenth/f-{int(f)}-{seg}.npz")["arr_0"]
            state_mat_seg[:, i*60:(i+1)*60] = cols
        sub_state_matrices.append(state_mat_seg)

    # Labels (assumed one-hot)
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    hard_y_train = np.argmax(labels_train, axis=1)
    hard_y_val   = np.argmax(labels_val, axis=1)
    hard_y_test  = np.argmax(labels_test, axis=1)

    num_train = labels_train.shape[0]
    num_val   = labels_val.shape[0]

    # Split full and sub matrices
    train_full = state_matrix_full[:num_train]
    val_full   = state_matrix_full[num_train:num_train + num_val]
    test_full  = state_matrix_full[num_train + num_val:]

    sub_train_list = [sub[:num_train] for sub in sub_state_matrices]
    sub_val_list   = [sub[num_train:num_train + num_val] for sub in sub_state_matrices]
    sub_test_list  = [sub[num_train + num_val:] for sub in sub_state_matrices]

    # Train 10 sub-models (teacher ensemble)
    W_list = []
    scaler_sub_list = []
    for k in range(num_tenths):
        print(f"Training sub-model {k+1}/{num_tenths}")
        scaler_k = StandardScaler()
        X_train_k = scaler_k.fit_transform(sub_train_list[k])
        X_val_k   = scaler_k.transform(sub_val_list[k])

        outputs_arr_k = []
        for lam in lambda_values:

            outputs_k = ridge_regression_fast(X_train_k, labels_train, X_val_k, labels_val, lam)
            outputs_arr_k.append(outputs_k)

        outputs_arr_k = np.array(outputs_arr_k)
        best_idx_k = np.argmax(outputs_arr_k[:, 2])
        best_lam_k = outputs_arr_k[best_idx_k, 0]
        best_val_k = outputs_arr_k[best_idx_k, 2]

        print(f"  Best λ = {best_lam_k:.1e}, val accuracy = {best_val_k:.4f}")

        W_k = compute_ridge_weights(X_train_k, labels_train, best_lam_k)
        W_list.append(W_k)
        scaler_sub_list.append(scaler_k)

    # Generate soft targets from teacher ensemble
    probs_train_list = []
    probs_val_list   = []
    probs_test_list  = []
    for k in range(num_tenths):
        # Train
        logits_train_k = predict_logits(scaler_sub_list[k].transform(sub_train_list[k]), W_list[k])
        probs_train_list.append(softmax(logits_train_k, axis=1))
        
        # Val (for ensemble baseline)
        logits_val_k = predict_logits(scaler_sub_list[k].transform(sub_val_list[k]), W_list[k])
        probs_val_list.append(softmax(logits_val_k, axis=1))
        
        # Test (for ensemble baseline)
        logits_test_k = predict_logits(scaler_sub_list[k].transform(sub_test_list[k]), W_list[k])
        probs_test_list.append(softmax(logits_test_k, axis=1))

    soft_Y_train = np.mean(probs_train_list, axis=0)

    # Ensemble baseline performance
    ensemble_val_acc = accuracy_score(hard_y_val, np.argmax(np.mean(probs_val_list, axis=0), axis=1))
    ensemble_test_acc = accuracy_score(hard_y_test, np.argmax(np.mean(probs_test_list, axis=0), axis=1))
    print(f"Teacher ensemble val accuracy: {ensemble_val_acc:.4f}")
    print(f"Teacher ensemble test accuracy: {ensemble_test_acc:.4f}")

    # Train distilled student on full 1 s features with soft targets
    scaler_full = StandardScaler()
    X_train_full_std = scaler_full.fit_transform(train_full)
    X_val_full_std   = scaler_full.transform(val_full)
    X_test_full_std  = scaler_full.transform(test_full)

    outputs_arr = []
    for lam in lambda_values:

        outputs = ridge_regression_fast(X_train_full_std, soft_Y_train, X_val_full_std, labels_val, lam, hard_y_train=hard_y_train)
        outputs_arr.append(outputs)

    outputs_arr = np.array(outputs_arr)
    best_idx = np.argmax(outputs_arr[:, 2])
    best_lam = outputs_arr[best_idx, 0]
    best_val = outputs_arr[best_idx, 2]
    print(f"Distilled best λ = {best_lam:.1e}, val accuracy = {best_val:.4f}")

    # Final evaluation on test
    W_final = compute_ridge_weights(X_train_full_std, soft_Y_train, best_lam)
    test_logits = predict_logits(X_test_full_std, W_final)
    test_hats = np.argmax(test_logits, axis=1)
    test_acc = accuracy_score(hard_y_test, test_hats)
    test_prec = precision_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_rec = recall_score(hard_y_test, test_hats, average='macro', zero_division=0)
    test_f1 = f1_score(hard_y_test, test_hats, average='macro', zero_division=0)

    # print(f"Distilled model test accuracy: {test_acc:.4f}")
    # print(f"Distilled model test precision/recall/f1: {test_prec:.4f}/{test_rec:.4f}/{test_f1:.4f}")
    print(f"\nDistilled student test accuracy:  {test_acc:.4f}")
    print(f"Distilled student test precision: {test_prec:.4f}")
    print(f"Distilled student test recall:    {test_rec:.4f}")
    print(f"Distilled student test F1:        {test_f1:.4f}")