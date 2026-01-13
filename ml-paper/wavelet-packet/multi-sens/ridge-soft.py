import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
plt.style.use('ggplot')


def softmax(logits):
    """Numerically stable softmax."""
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    A = X_train.T @ X_train + lam * I
    b = X_train.T @ Y_train
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b

def ridge_regression_fast(
    X_train, Y_train, X_eval, Y_eval, lam,
    bootstrap_iters=5, beta=0.95
):
    """
    Modified Ridge regression with optional iterative soft bootstrapping.
    
    - If bootstrap_iters=0 → original hard-label behavior.
    - If bootstrap_iters>0 → "smart" soft-label bootstrapping:
        Iteratively mix the model's confident predictions on the training data
        back into the targets. This encourages consistency and robustness to
        noisy samples (e.g., audio recordings corrupted by noise that make
        features unreliable/outliers).
        
        Noisy samples tend to get low-confidence predictions → their targets
        are gradually softened toward the model's consistent view, effectively
        down-weighting them. This acts as a form of self-distillation / robust
        regularization without needing extra data or complex losses.
    """
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Assume Y_train/Y_eval are one-hot (N x C) as in your loading code
    Y_train_oh = Y_train.astype(np.float64)
    y_train_true = np.argmax(Y_train_oh, axis=1)
    y_eval_true  = np.argmax(Y_eval, axis=1)

    # -----------------------------
    # Bootstrapping loop (smart soft labels)
    # -----------------------------
    current_targets = Y_train_oh.copy()

    for it in range(bootstrap_iters):
        # Fit on current (soft) targets
        W = ridge_closed_form(X_train_b, current_targets, lam)

        # Get soft predictions on training data
        logits_train = X_train_b @ W
        proba_train = softmax(logits_train)

        # Update targets: convex combination (beta trusts original labels)
        current_targets = beta * Y_train_oh + (1.0 - beta) * proba_train

    # Final fit on the refined soft targets
    W = ridge_closed_form(X_train_b, current_targets, lam)

    # -----------------------------
    # Predictions (using final model)
    # -----------------------------
    logits_train = X_train_b @ W
    y_train_hats = np.argmax(logits_train, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    logits_eval = X_eval_b @ W
    y_eval_hats = np.argmax(logits_eval, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    
    # Include lambda (and optionally bootstrap params) for easy analysis
    results = np.array([
        lam, bootstrap_iters, beta,
        train_accuracy, accuracy, precision, recall, f1
    ], dtype=np.float64)
    
    return results



if __name__ == '__main__':

    a = 0.9
    mu = 1.0 
    u_dc = 1.0

    lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
    
    f_values = np.linspace(1000, 50000, 36)

    state_matrix = np.zeros((10510, len(f_values) * 368))
    
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*368:(i+1)*368] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    train_state = state_matrix[:len(labels_train)]
    val_state   = state_matrix[len(labels_train):]

    # scale PER FOLD (no leakage)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std   = scaler.transform(val_state)

    # -----------------------------
    # Hyperparameters for soft bootstrapping
    # -----------------------------
    bootstrap_iters = 5      # 3–10 is typical; 0 = disable (original hard labels)
    beta = 0.95              # 0.9–0.99; higher = more trust in original labels

    # evaluate all lambdas (parallel)
    outputs = Parallel(
        n_jobs=64,
        verbose=1,
        backend="multiprocessing"
    )(
        delayed(ridge_regression_fast)(
            X_train_std, labels_train,
            X_val_std,  labels_val,
            lam,
            bootstrap_iters=bootstrap_iters,
            beta=beta
        )
        for lam in lambda_values
    )

    outputs_arr = np.vstack(outputs)

    # plot the results
    lambdas   = outputs_arr[:, 0]
    train_acc = outputs_arr[:, 3] * 100
    test_acc  = outputs_arr[:, 4] * 100

    idx_best = np.argmax(test_acc)

    best_test   = test_acc[idx_best]
    best_train  = train_acc[idx_best]
    best_lambda = lambdas[idx_best]

    # std across λ (for uncertainty band)
    std_test = np.std(test_acc)
    std_train = np.std(train_acc)

    # -----------------------------
    # Plot with error bars
    # -----------------------------
    plt.figure(figsize=(16, 8))

    plt.plot(
        lambda_values, test_acc,
        marker='o',
        linewidth=3,
        label="Validation Accuracy"
    )

    plt.plot(
        lambda_values, train_acc,
        marker='s',
        linewidth=3,
        linestyle='--',
        label="Training Accuracy"
    )

    # highlight best lambda
    plt.scatter(
        best_lambda,
        best_test,
        s=200,
        marker='*',
        zorder=5,
        label=(
            f"Best val = {best_test:.2f}%\n"
            f"λ = {best_lambda:.1e}"
        )
    )

    # Formatting
    plt.xscale('log')
    plt.xlabel("Ridge $\\lambda$", fontweight='bold', fontsize=20)
    plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()

    plt.savefig(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/lambda-optimization-a-{a:.2f}-u_dc-{u_dc:.2f}-soft.png", dpi=300)
    plt.close()

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/accuracy"
    os.makedirs(results_dir, exist_ok=True)

    # Save raw metrics
    np.savetxt(
        f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-{mu:.2e}-soft.txt",
        outputs_arr,
        fmt=["%.0e", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
        header="lambda,bootstrap_iters,beta,train_acc,val_acc,precision,recall,f1",
        comments=""
    )

    # # delete cols
    # for i, f in enumerate(f_values):
    #     os.remove(f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/f-{int(f)}.npz")

    # err_out_dir = Path(
    #     f"/scratch/almo2783/scratch/ml-paper/wavelet-packet/multi-sens/results/a-{a:.2f}-u_dc-{u_dc:.2f}/err-out"
    # )

    # if err_out_dir.exists():
    #     shutil.rmtree(err_out_dir)