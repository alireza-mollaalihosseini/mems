import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# ----------------------------
# numba RK4 for 2D Hopf (x,y)
# ----------------------------

@njit(fastmath=True)
def hopf_rhs(x, y, lam, alpha, beta, f_x):
    """
    RHS of the forced Hopf normal form:
      dx = lam*x - y + r2*(alpha*x - beta*y) + f_x
      dy = lam*y + x + r2*(alpha*y + beta*x)
    where r2 = x^2 + y^2
    """
    r2 = x * x + y * y
    dx = lam * x - y + r2 * (alpha * x - beta * y) + f_x
    dy = lam * y + x + r2 * (alpha * y + beta * x)
    return dx, dy


@njit(fastmath=True)
def rk4_step_hopf_inplace(y, h, lam, alpha, beta, k1, k2, k3, k4, y_temp):
    """
    RK4 step without external forcing (f_x = 0)
    y is length-2 array [x, y]
    """
    x = y[0]
    yy = y[1]

    # k1
    k1x, k1y = hopf_rhs(x, yy, lam, alpha, beta, 0.0)
    k1[0] = k1x
    k1[1] = k1y

    # k2
    y_temp[0] = x + 0.5 * h * k1[0]
    y_temp[1] = yy + 0.5 * h * k1[1]
    k2x, k2y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, 0.0)
    k2[0] = k2x
    k2[1] = k2y

    # k3
    y_temp[0] = x + 0.5 * h * k2[0]
    y_temp[1] = yy + 0.5 * h * k2[1]
    k3x, k3y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, 0.0)
    k3[0] = k3x
    k3[1] = k3y

    # k4
    y_temp[0] = x + h * k3[0]
    y_temp[1] = yy + h * k3[1]
    k4x, k4y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, 0.0)
    k4[0] = k4x
    k4[1] = k4y

    # update y
    y[0] += (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    y[1] += (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])


@njit(fastmath=True)
def rk4_step_hopf_inplace_with_force(y, h, lam, alpha, beta, k1, k2, k3, k4, y_temp, f_x):
    """
    RK4 step with external forcing f_x applied to dx.
    y: length-2 array [x, y]
    """
    x = y[0]
    yy = y[1]

    # k1
    k1x, k1y = hopf_rhs(x, yy, lam, alpha, beta, f_x)
    k1[0] = k1x
    k1[1] = k1y

    # k2
    y_temp[0] = x + 0.5 * h * k1[0]
    y_temp[1] = yy + 0.5 * h * k1[1]
    # approximate mid-step forcing by same f_x (we pass f_x constant within a sample step)
    k2x, k2y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, f_x)
    k2[0] = k2x
    k2[1] = k2y

    # k3
    y_temp[0] = x + 0.5 * h * k2[0]
    y_temp[1] = yy + 0.5 * h * k2[1]
    k3x, k3y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, f_x)
    k3[0] = k3x
    k3[1] = k3y

    # k4
    y_temp[0] = x + h * k3[0]
    y_temp[1] = yy + h * k3[1]
    k4x, k4y = hopf_rhs(y_temp[0], y_temp[1], lam, alpha, beta, f_x)
    k4[0] = k4x
    k4[1] = k4y

    # update y
    y[0] += (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    y[1] += (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])


# ----------------------------
# Simulation helpers
# ----------------------------

@njit(fastmath=True)
def simulate_transient_hopf(N, h, lam, alpha, beta):
    """
    Run unforced transient to reach steady state (returns y_final length-2).
    """
    y = np.zeros(2, dtype=np.float64)
    # small initial condition
    y[0] = 1e-9
    y[1] = 0.0

    k1 = np.zeros(2, dtype=np.float64)
    k2 = np.zeros(2, dtype=np.float64)
    k3 = np.zeros(2, dtype=np.float64)
    k4 = np.zeros(2, dtype=np.float64)
    y_temp = np.zeros(2, dtype=np.float64)

    for k in range(N):
        rk4_step_hopf_inplace(y, h, lam, alpha, beta, k1, k2, k3, k4, y_temp)

    return y


@njit(fastmath=True)
def simulate_with_force_hopf(y, N, h, lam, alpha, beta, f_ext, buf_x_out):
    """
    Integrate the forced Hopf for N steps. Return buffer of x (real part) values.
    y: length-2 initial state (modified inplace).
    f_ext: length-N external forcing (real)
    buf_x_out: preallocated length-N output buffer (x values)
    """
    k1 = np.zeros(2, dtype=np.float64)
    k2 = np.zeros(2, dtype=np.float64)
    k3 = np.zeros(2, dtype=np.float64)
    k4 = np.zeros(2, dtype=np.float64)
    y_temp = np.zeros(2, dtype=np.float64)

    for k in range(N):
        f_x = f_ext[k]
        rk4_step_hopf_inplace_with_force(y, h, lam, alpha, beta, k1, k2, k3, k4, y_temp, f_x)
        buf_x_out[k] = y[0]   # store x (real part) as output (can also store z = x + i y if desired)

    return buf_x_out


# ----------------------------
# File processing and wrapper
# ----------------------------

def process_file(fname, y_final, N_force, h, lam, alpha, beta, mu, fft_len, fs):
    """
    Reads file fname, picks/resamples using idxs, filters, pads/truncates to N_force,
    runs the forced Hopf integration and returns FFT magnitudes (rfft).
    mu: force scaling (gain) applied to audio before feeding into RK4
    """
    data, sr = sf.read(fname)
    
    # prepare forcing buffer (1 second at fs -> N_force samples)
    signal_buf = np.zeros(N_force, dtype=np.float64)
    n_copy = min(len(data), N_force)
    signal_buf[:n_copy] = data[:n_copy]

    # scale forcing by mu (gain)
    signal_buf *= mu

    # initial state copy
    y0 = y_final.copy()  # length-2

    # output buffer for x
    buf_x_out = np.empty(N_force, dtype=np.float64)

    # run forced simulation
    u_x = simulate_with_force_hopf(y0, N_force, h, lam, alpha, beta, signal_buf, buf_x_out)

    # compute FFT (rfft) and return magnitudes (first fft_len components)
    fft_vals = np.fft.rfft(u_x)
    return np.abs(fft_vals[:fft_len]).astype(np.float32)


def build_state_matrix(train_file_list_path, val_file_list_path, test_file_list_path, alpha, beta, lam, mu, fs=44100):
    """
    Build state matrix using audio-driven Hopf oscillator.
    alpha, beta: complex cubic coefficient b = alpha + i beta
    lam: linear Hopf parameter
    mu: forcing gain (multiply audio by mu)
    fs: sampling rate used for forcing (44100)
    """

    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    n_files = len(filenames)

    # Simulation params for discrete integration
    dt = 1.0 / fs
    h = dt

    # For transient, run some cycles (e.g. 0.2 s) to reach steady state
    transient_seconds = 3000.0
    N_trans = int(transient_seconds * fs)

    # Force length = 1 second -> N_force = fs (samples)
    N_force = int(1.0 * fs)

    # FFT length we want to store
    fft_len = 22001

    # Initial transient to obtain steady state initial condition
    y_final = simulate_transient_hopf(N_trans, h, lam, alpha, beta)

    # run processing in parallel
    results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(process_file)(
            fname, y_final, N_force, h, lam, alpha, beta, mu, fft_len, fs
        )
        for fname in filenames
    )

    state_matrix = np.vstack(results)
    
    return state_matrix



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
    
    results = np.array([lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    
    return results


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, required=True, help='Value of lambda to process')
args = parser.parse_args()

# ----------------------------
# main
# ----------------------------
if __name__ == '__main__':

    lam   = args.lam
    alpha = -0.01
    beta  = 0.3
    mu    = 1e4 # 0.1 # 1.0
    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])

    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    # labels
    labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")
    labels_test   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)

    # -----------------------------
    # K-Fold Configuration
    # -----------------------------
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/results"
    os.makedirs(results_dir, exist_ok=True)

    n_lams = len(lambda_values)

    state_matrix = build_state_matrix(train_files, val_files, test_files, alpha, beta, lam, mu)

    # container: results[lambda_idx][fold] -> 6 metrics
    results_per_lambda = [ [] for _ in range(n_lams) ]

    for fold, (train_idx, test_idx) in enumerate(kf.split(state_matrix), start=1):

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
                lambda_value
            )
            for lambda_value in lambda_values
        )

        # -----------------------------
        # Save per-fold results
        # -----------------------------
        for i, (metrics) in enumerate(fold_outputs):

            lambda_value = lambda_values[i]

            # store for mean/std later
            results_per_lambda[i].append(metrics)

            # Save raw metrics
            np.savetxt(
                f"{results_dir}/fold_results-alpha-{alpha:.2f}-beta-{beta:.2f}-lam-{lam:.2f}-mu-{mu:.0e}-lambda-ridge-{lambda_value:.1e}-fold-{fold}.txt",
                metrics.reshape(1,-1),
                fmt="%.6f"
            )
