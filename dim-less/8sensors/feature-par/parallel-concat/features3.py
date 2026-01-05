import itertools
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)


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

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)
    
    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results, conf_matrix


def top_features(W, top_k=10):
    # Exclude bias
    W_no_bias = W[:-1, :]

    feature_importance = {}
    for c in range(W_no_bias.shape[1]):
        importance = np.abs(W_no_bias[:, c])
        top_idx = np.argsort(importance)[::-1][:top_k]
        for idx in top_idx:
            if idx not in feature_importance:
                feature_importance[idx] = importance[idx]
            else:
                feature_importance[idx] = max(feature_importance[idx], importance[idx])

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_idx_final = np.array([idx for idx, _ in sorted_features[:top_k]])
    
    return top_idx_final


def process_top_k_and_sensors(top_k, lambda_grid, results_dir, sensor_subset,
                              sensor_data, sensor_weights,
                              labels_train, labels_test, labels_val):
    selected_train = []
    selected_test  = []
    selected_val   = []

    for sensor in sensor_subset:
        idx = top_features(sensor_weights[sensor], top_k=top_k)
        selected_train.append(sensor_data[sensor]["train"][:, idx])
        selected_test.append(sensor_data[sensor]["test"][:, idx])
        selected_val.append(sensor_data[sensor]["val"][:, idx])

    # Concatenate features from chosen sensors
    X_train = np.concatenate(selected_train, axis=1)
    X_test  = np.concatenate(selected_test, axis=1)
    X_val   = np.concatenate(selected_val, axis=1)

    # Standardize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_val   = scaler.transform(X_val)

    best_val_acc = -np.inf
    best_lambda  = None
    best_results_val, best_cm_val = None, None
    best_results_test, best_cm_test = None, None

    # Try all candidate lambdas
    for lam in lambda_grid:
        results_val, cm_val = ridge_regression_fast(X_train, labels_train, X_val, labels_val, lam)
        val_acc = results_val[2]  # validation accuracy

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lambda  = lam
            best_results_val, best_cm_val = results_val, cm_val

            # Evaluate test with same λ
            results_test, cm_test = ridge_regression_fast(X_train, labels_train, X_test, labels_test, lam)
            best_results_test, best_cm_test = results_test, cm_test

    # Name subset for saving
    subset_name = "-".join(map(str, sensor_subset))

    # Save best results
    np.savetxt(f"{results_dir}/results_test-sensors-{subset_name}-topk-{top_k}-lambda-{best_lambda}.txt",
               best_results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_test-sensors-{subset_name}-topk-{top_k}-lambda-{best_lambda}.txt",
               best_cm_test, fmt="%.5f")

    np.savetxt(f"{results_dir}/results_val-sensors-{subset_name}-topk-{top_k}-lambda-{best_lambda}.txt",
               best_results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"{results_dir}/conf_matrix_val-sensors-{subset_name}-topk-{top_k}-lambda-{best_lambda}.txt",
               best_cm_val, fmt="%.5f")


if __name__ == "__main__":
    # lambda_value = 1e4
    lambda_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000]
    mu = 1.0

    # --- Load full matrices ONCE ---
    X_train1_full = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val1_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test1_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_test-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train2_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_train-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_val2_full    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_val-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_test2_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_test-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0']

    X_train3_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_train-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_val3_full    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_val-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']
    X_test3_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_test-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0']

    X_train4_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_train-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val4_full    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_val-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test4_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_test-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train5_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_train-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    X_val5_full    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_val-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']
    X_test5_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_test-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0']

    X_train6_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_train-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_val6_full    = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_val-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']
    X_test6_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_test-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0']

    X_train7_full = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_val7_full   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']
    X_test7_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_test-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0']

    # Store all sensor data in one dict
    sensor_data = {
        1: {"train": X_train1_full, "val": X_val1_full, "test": X_test1_full},
        2: {"train": X_train2_full, "val": X_val2_full, "test": X_test2_full},
        3: {"train": X_train3_full, "val": X_val3_full, "test": X_test3_full},
        4: {"train": X_train4_full, "val": X_val4_full, "test": X_test4_full},
        5: {"train": X_train5_full, "val": X_val5_full, "test": X_test5_full},
        6: {"train": X_train6_full, "val": X_val6_full, "test": X_test6_full},
        7: {"train": X_train7_full, "val": X_val7_full, "test": X_test7_full},
    }

    # Load the weights
    weights1 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/weights/weights-a-0.16-lambda-10000.0.npz")['arr_0']
    weights2 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/weights/weights-a-0.04-lambda-10000.0.npz")['arr_0']
    weights3 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/weights/weights-a-0.02-lambda-10000.0.npz")['arr_0']
    weights4 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/weights/weights-a-0.6-lambda-10000.0.npz")['arr_0']
    weights5 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/weights/weights-a-0.06-lambda-10000.0.npz")['arr_0']
    weights6 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']
    weights7 = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']

    sensor_weights = {
        1: weights1,
        2: weights2,
        3: weights3,
        4: weights4,
        5: weights5,
        6: weights6,
        7: weights7,
    }

    # Labels
    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/lam-opt"
    os.makedirs(results_dir, exist_ok=True)

    sensors = list(sensor_data.keys())

    # Generate only pairs, triplets, quartets, and all 7
    subsets = []
    subsets.extend(itertools.combinations(sensors, 1))
    # subsets.extend(itertools.combinations(sensors, 2))  # pairs
    # subsets.extend(itertools.combinations(sensors, 3))  # triplets
    subsets.extend(itertools.combinations(sensors, 4))  # quartets
    subsets.extend(itertools.combinations(sensors, 5))  # quartets
    subsets.extend(itertools.combinations(sensors, 6))  # quartets
    # subsets.append(tuple(sensors))                      # all 7

    Parallel(n_jobs=8, verbose=10)(
        delayed(process_top_k_and_sensors)(
            top_k, lambda_grid, results_dir, subset, sensor_data, sensor_weights,
            labels_train, labels_test, labels_val
        )
        for top_k in top_k_values
        for subset in subsets
    )
