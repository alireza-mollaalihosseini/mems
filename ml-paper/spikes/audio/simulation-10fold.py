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


def process_file(fname, spike_threshold=0.4, refractory_period=50):
    data, _ = sf.read(fname)
    
    # Normalize between 0 and 1

    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # -------------------------
    # Spike generation
    # -------------------------
    spike_indices = []
    last_spike = -refractory_period

    for i, value in enumerate(data):
        if value > spike_threshold and (i - last_spike) >= refractory_period:
            spike_indices.append(i)
            last_spike = i

    spike_indices = np.array(spike_indices, dtype=np.int64)

    # -------------------------
    # Event-driven reference points
    # -------------------------
    N_POINTS = 100
    N_SPIKES = 12

    # Only times where at least 10 spikes already occurred
    valid_times = spike_indices[N_SPIKES - 1:]

    # If signal is short, guard against failure
    if len(valid_times) < N_POINTS:
        raise ValueError("Not enough spikes to generate 100 event-driven points")

    # Uniformly sample 100 reference points from valid event times
    ref_points = np.linspace(
        0, len(valid_times) - 1, N_POINTS, dtype=np.int64
    )
    ref_points = valid_times[ref_points]

    # -------------------------
    # Last-10-spike encoding
    # -------------------------
    spike_time_matrix = np.zeros((N_POINTS, N_SPIKES), dtype=np.int64)

    for i, t in enumerate(ref_points):
        # Find spikes before or at time t
        idx = np.searchsorted(spike_indices, t, side="right")

        last_spikes = spike_indices[idx - N_SPIKES : idx]
        spike_time_matrix[i] = t - last_spikes

    return spike_time_matrix.flatten()



def build_state_matrices(train_file_list_path, val_file_list_path, test_file_list_path, spike_threshold, refractory_period):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    all_filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    
    # Single parallel call to process all files (reduces joblib overhead)
    results = Parallel(n_jobs=64, verbose=1, backend='multiprocessing')(
        delayed(process_file)(fname, spike_threshold, refractory_period)
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

    lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6]

    spike_threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    refractory_period_values = [5, 10, 15, 20, 25, 30]

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/spikes/audio/results"
    os.makedirs(results_dir, exist_ok=True)

    for spike_threshold in spike_threshold_values:
        for refractory_period in refractory_period_values:
            print(f"\n=== Spike Threshold: {spike_threshold} | Refractory Period: {refractory_period} ===")

            state_matrix = build_state_matrices(train_files, val_files, test_files, spike_threshold, refractory_period)

            # -----------------------------
            # K-Fold Configuration
            # -----------------------------
            n_splits = 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            n_lams = len(lambda_values)
            
            # container: results[lambda_idx][fold] -> 6 metrics
            results_per_lambda = [ [] for _ in range(n_lams) ]
            best_lambdas = []
            best_train_accuracies = []
            best_test_accuracies = []

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

                fold_outputs_arr = np.vstack(fold_outputs)

                best_idx = np.argmax(fold_outputs_arr[:, 2])
                best_lam = fold_outputs_arr[best_idx, 0]
                best_train_acc = fold_outputs_arr[best_idx, 1]
                best_test_acc = fold_outputs_arr[best_idx, 2]

                best_lambdas.append(best_lam)
                best_train_accuracies.append(best_train_acc)
                best_test_accuracies.append(best_test_acc)

                # # -----------------------------
                # # Save per-fold results
                # # -----------------------------
                # for i, (metrics) in enumerate(fold_outputs):

                #     lam = lambda_values[i]

                #     # store for mean/std later
                #     results_per_lambda[i].append(metrics)

                #     # Save raw metrics
                #     np.savetxt(
                #         f"{results_dir}/fold_results-lambda-{lam:.1e}-fold-{fold}.txt",
                #         metrics.reshape(1,-1),
                #         fmt="%.6f"
                #     )

            # print average best train and test accuracies
            print("\n--- Summary over folds ---")
            avg_best_train_acc = np.mean(best_train_accuracies)
            avg_best_test_acc = np.mean(best_test_accuracies)
            print(f"Average Best Train Accuracy: {avg_best_train_acc*100:.2f}%")
            print(f"Average Best Test Accuracy: {avg_best_test_acc*100:.2f}%")

            # # -----------------------------
            # # Compute Mean ± Std over folds
            # # -----------------------------
            # for i, lam in enumerate(lambda_values):

            #     fold_data = np.vstack(results_per_lambda[i])   # shape: (n_folds,6)

            #     mean = fold_data.mean(axis=0)
            #     std  = fold_data.std(axis=0)

            #     summary = np.vstack([
            #         mean,
            #         std
            #     ])

            #     np.savetxt(
            #         f"{results_dir}/summary-lambda-{lam:.1e}.txt",
            #         summary,
            #         header="ROW1=MEAN  ROW2=STD\n"
            #             "COLS=[lambda, train_acc, test_acc, precision, recall, f1]",
            #         fmt="%.8f"
            #     )

            # print("\n✅ K-Fold cross-validation finished successfully.")