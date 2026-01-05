import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


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

    a = 0.44
    mu = 1.0 
    u_dc = 0.4

    # lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
    # f_values = np.array([8000, 9000, 10000, 11000, 12000])
    # f_values = np.linspace(8000, 30000, 201)
    f_values = np.arange(1000, 50000, 50)

    state_matrix = np.zeros((10510, len(f_values) * 16))
    
    for i, f in enumerate(f_values):
        cols = np.load(f"/scratch/almo2783/scratch/ml-paper/multi-sens/test/results/f-{int(f)}.npz")["arr_0"]
        state_matrix[:, i*16:(i+1)*16] = cols

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    train_state = state_matrix[:len(labels_train)]
    val_state   = state_matrix[len(labels_train):]

    # scale PER FOLD (no leakage)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std   = scaler.transform(val_state)

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

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/multi-sens/test/accuracy"
    os.makedirs(results_dir, exist_ok=True)

    # Save raw metrics
    np.savetxt(
        f"{results_dir}/results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-{mu:.2e}.txt",
        outputs_arr,
        fmt=["%.0e", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
        header="lambda,train_acc,val_acc,precision,recall,f1",
        comments=""
    )