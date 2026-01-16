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



if __name__ == '__main__':

    a = 0.9
    mu = 1.0 
    u_dc = 1.0

    # lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
    # f_values = np.linspace(1000, 50000, 101)
    f_values = np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
       42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
       38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
        4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
       38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
       40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
       21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
       35790])
    f_values = np.sort(f_values)

    state_matrix = np.zeros((10910, len(f_values) * 60))
    
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/100/results/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*60:(i+1)*60] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    lebels_test  = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")

    train_state = state_matrix[:len(labels_train)]
    val_state   = state_matrix[len(labels_train): len(labels_train) + len(labels_val)]
    test_state  = state_matrix[len(labels_train) + len(labels_val): ]

    # scale PER FOLD (no leakage)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std   = scaler.transform(val_state)
    X_test_std  = scaler.transform(test_state)

    # evaluate all lambdas (parallel)
    outputs = Parallel(
        n_jobs=64,
        verbose=1,
        backend="multiprocessing"
    )(
        delayed(ridge_regression_fast)(
            X_train_std, labels_train,
            X_val_std,  labels_val,
            lam
        )
        for lam in lambda_values
    )

    outputs_arr = np.vstack(outputs)

    # plot the results
    lambdas   = outputs_arr[:, 0]
    train_acc = outputs_arr[:, 1] * 100
    test_acc  = outputs_arr[:, 2] * 100

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

    plt.savefig(f"lambda-optimization-a-{a:.2f}-u_dc-{u_dc:.2f}-top-64.png", dpi=300)
    plt.close()

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/multi-sens/100"
    os.makedirs(results_dir, exist_ok=True)

    # Save raw metrics
    np.savetxt(
        f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-{mu:.2e}-top-64.txt",
        outputs_arr,
        fmt=["%.0e", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
        header="lambda,train_acc,val_acc,precision,recall,f1",
        comments=""
    )

    # ridge regression best model on test set
    best_results = ridge_regression_fast(X_train_std, labels_train,
                                        X_test_std,  lebels_test,
                                        best_lambda)

    print("Best model on TEST set:")
    print(f"Lambda: {best_results[0]:.1e}")
    print(f"Train Accuracy: {best_results[1]*100:.2f}%")
    print(f"Test Accuracy: {best_results[2]*100:.2f}%")
    print(f"Precision: {best_results[3]*100:.2f}%")
    print(f"Recall: {best_results[4]*100:.2f}%")
    print(f"F1 Score: {best_results[5]*100:.2f}%")

    # # delete cols
    # for i, f in enumerate(f_values):
    #     os.remove(f"/scratch/almo2783/scratch/ml-paper/multi-sens/test/results/f-{int(f)}.npz")

    # err_out_dir = Path(
    #     f"/scratch/almo2783/scratch/ml-paper/multi-sens/test/results/err-out"
    # )

    # if err_out_dir.exists():
    #     shutil.rmtree(err_out_dir)