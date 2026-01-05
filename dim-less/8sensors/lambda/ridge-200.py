import sys
import time
import os
import numpy as np
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


def process_lambda_value(X_train, X_val, labels_train, labels_val, lambda_value, a_value, u_dc_value, results_dir):

    results, conf_matrix = ridge_regression_fast(
        X_train, labels_train, X_val, labels_val, lambda_value, a_value, u_dc_value
    )

    # Save results
    np.savetxt(f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-lambda-{lambda_value:.1e}-200.txt",
               results.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-lambda-{lambda_value:.1e}-200.txt",
               conf_matrix, fmt="%.5f")


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    lambda_values = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18]
    mu = 1.0

    # Load training, validation
    X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}.npz")['arr_0']
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    extracted_features_idx = np.array([    0,     1,     2,     3,     4,     9,    10,    11,    12,
                                           13,    14,    15,    16,    17,    18,    19,    20,    21,
                                           22,    23,    24,    25,    26,    27,    28,    29,    30,
                                           31,    32,    33,    34,    35,    36,    37,    38,    39,
                                           40,    41,    42,    45,    46,    47,    48,    49,    50,
                                           51,    52,    53,    54,    55,    56,    57,    58,    59,
                                           60,    63,    65,    67,    68,    69,    70,    71,    74,
                                           75,    77,    78,    79,    80,    82,    83,    84,    90,
                                           92,    97,    98,    99,   100,   101,   102,   103,   105,
                                           106,   108,   110,   111,   112,   113,   114,   115,   116,
                                           117,   121,   123,   132,   134,   143,   146,   147,   148,
                                           149,   150,   151,   154,   155,   156,   159,   170,   175,
                                           192,   193,   194,   196,   197,   198,   200,   202,   210,
                                           219,   220,   227,   238,   239,   247,   249,   267,   287,
                                           289,   290,   291,   293,   300,   301,   333,   340,   345,
                                           369,   370,   381,   382,   419,   424,   439,   440,   497,
                                           521,   591,   657,   658,   699,   804,   989,   997,  1228,
                                           1229,  1273,  1274,  1275,  1276,  1277,  1278,  1279,  1290,
                                           1411,  1440,  1471,  1472,  1473,  1474,  1500,  1501,  1582,
                                           2500,  2501,  2866,  2867,  2999,  3000,  3175,  3668,  3879,
                                           4000,  4150,  5001,  5200,  5948,  5949,  5950,  5999,  6000,
                                           6049,  6249,  6933,  7500,  8000,  8001,  8667,  9000, 14999,
                                           15000, 17503])

    X_train = X_train[:, extracted_features_idx]
    X_val   = X_val[:, extracted_features_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/lambda/results"
    os.makedirs(results_dir, exist_ok=True)

    # Run in parallel
    Parallel(n_jobs=-1, backend="threading", verbose=10)(  # adjust n_jobs depending on your cluster
        delayed(process_lambda_value)(
            X_train, X_val, labels_train, labels_val,
            lambda_value, a_value, u_dc_value, results_dir
        )
        for idx, lambda_value in enumerate(lambda_values)
    )
