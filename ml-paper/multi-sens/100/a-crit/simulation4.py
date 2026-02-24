import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from numba import njit

@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x):
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy + c5 * f_x
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy + c5 * f_x
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy + c5 * f_x
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy + c5 * f_x
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def simulate_transient(N, h, c1, c2, c3, c4, phi_dc, a):

  y  = np.zeros(4)
  y[0] = 1e-9
  k1 = np.zeros(4)
  k2 = np.zeros(4)
  k3 = np.zeros(4)
  k4 = np.zeros(4)
  y_temp = np.zeros(4)

  for k in range(N):
    rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)

  return y


@njit(fastmath=True)
def simulate_with_force(y, N, h, c1, c2, c3, c4, c5, phi_dc, a, f_ext):

  k1 = np.zeros(4)
  k2 = np.zeros(4)
  k3 = np.zeros(4)
  k4 = np.zeros(4)
  y_temp = np.zeros(4)
  buf_u_ac = np.empty(N)

  for k in range(N):
    f_x = f_ext[k]
    rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x)
    buf_u_ac[k] = y[3]

  return buf_u_ac


def extract_features(u_ac_buf):
    du = np.diff(u_ac_buf)
    d2u = np.diff(u_ac_buf, n=2)

    abs_u = np.abs(u_ac_buf)
    abs_du = np.abs(du)
    abs_d2u = np.abs(d2u)

    rms_u = np.sqrt(np.mean(u_ac_buf**2))
    rms_du = np.sqrt(np.mean(du**2))
    rms_d2u = np.sqrt(np.mean(d2u**2))

    peak_u = np.max(abs_u)
    peak_du = np.max(abs_du)
    peak_d2u = np.max(abs_d2u)

    mean_abs_u = np.mean(abs_u)
    mean_abs_du = np.mean(abs_du)
    mean_abs_d2u = np.mean(abs_d2u)

    crest_u = peak_u / rms_u if rms_u > 0 else 0.0
    crest_du = peak_du / rms_du if rms_du > 0 else 0.0
    crest_d2u = peak_d2u / rms_d2u if rms_d2u > 0 else 0.0

    shape_u = rms_u / mean_abs_u if mean_abs_u > 0 else 0.0
    shape_du = rms_du / mean_abs_du if mean_abs_du > 0 else 0.0
    shape_d2u = rms_d2u / mean_abs_d2u if mean_abs_d2u > 0 else 0.0

    impulse_u = peak_u / mean_abs_u if mean_abs_u > 0 else 0.0
    impulse_du = peak_du / mean_abs_du if mean_abs_du > 0 else 0.0
    impulse_d2u = peak_d2u / mean_abs_d2u if mean_abs_d2u > 0 else 0.0

    sqrt_abs_mean_u = np.mean(np.sqrt(abs_u))
    sqrt_abs_mean_du = np.mean(np.sqrt(abs_du))
    sqrt_abs_mean_d2u = np.mean(np.sqrt(abs_d2u))

    clearance_u = peak_u / (sqrt_abs_mean_u ** 2) if sqrt_abs_mean_u > 0 else 0.0
    clearance_du = peak_du / (sqrt_abs_mean_du ** 2) if sqrt_abs_mean_du > 0 else 0.0
    clearance_d2u = peak_d2u / (sqrt_abs_mean_d2u ** 2) if sqrt_abs_mean_d2u > 0 else 0.0

    skew_u = skew(u_ac_buf)
    kurt_u = kurtosis(u_ac_buf)
    skew_du = skew(du)
    kurt_du = kurtosis(du)
    skew_d2u = skew(d2u)
    kurt_d2u = kurtosis(d2u)

    median_u = np.median(u_ac_buf)
    p25_u, p75_u = np.percentile(u_ac_buf, [25, 75])
    iqr_u = p75_u - p25_u

    median_du = np.median(du)
    p25_du, p75_du = np.percentile(du, [25, 75])
    iqr_du = p75_du - p25_du

    median_d2u = np.median(d2u)
    p25_d2u, p75_d2u = np.percentile(d2u, [25, 75])
    iqr_d2u = p75_d2u - p25_d2u

    norm_argmax_u = np.argmax(abs_u) / len(u_ac_buf)
    norm_argmax_du = np.argmax(abs_du) / len(du)
    norm_argmax_d2u = np.argmax(abs_d2u) / len(d2u)

    zcr_u = np.sum(np.diff(np.signbit(u_ac_buf))) / (len(u_ac_buf) - 1.0)
    zcr_du = np.sum(np.diff(np.signbit(du))) / (len(du) - 1.0)
    zcr_d2u = np.sum(np.diff(np.signbit(d2u))) / (len(d2u) - 1.0)

    ptp_u = np.ptp(u_ac_buf)
    ptp_du = np.ptp(du)
    ptp_d2u = np.ptp(d2u)

    feats = [
        np.mean(u_ac_buf), np.std(u_ac_buf), median_u, p25_u, p75_u, iqr_u, mean_abs_u,
        np.mean(du), np.std(du), median_du, p25_du, p75_du, iqr_du, mean_abs_du,
        np.mean(d2u), np.std(d2u), median_d2u, p25_d2u, p75_d2u, iqr_d2u, mean_abs_d2u,
        np.max(u_ac_buf), np.argmax(u_ac_buf), peak_u, norm_argmax_u,
        np.max(du), np.argmax(du), peak_du, norm_argmax_du,
        np.max(d2u), np.argmax(d2u), peak_d2u, norm_argmax_d2u,
        rms_u, rms_du, rms_d2u,
        ptp_u, ptp_du, ptp_d2u,
        crest_u, crest_du, crest_d2u,
        shape_u, shape_du, shape_d2u,
        impulse_u, impulse_du, impulse_d2u,
        clearance_u, clearance_du, clearance_d2u,
        skew_u, kurt_u, skew_du, kurt_du, skew_d2u, kurt_d2u,
        zcr_u, zcr_du, zcr_d2u,
    ]

    return np.array(feats, dtype=np.float32)


def compute_transient(parameters, mu=1.0):
    f, a, u_dc = parameters
    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 50.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0
    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / u_max
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max

    y_final = simulate_transient(50000000, h, c1, c2, c3, c4, phi_dc, a)
    return (y_final, h, c1, c2, c3, c4, c5, phi_dc, a)


def process_one_file(fname, precomp_list):
    data, sr = sf.read(fname)
    
    new_len = 1000000
    frac = new_len / sr
    idxs_len = int(len(data) * frac)
    idxs = (np.arange(idxs_len) / frac).astype(np.int64)
    signal = data[idxs]

    signal = np.asarray(signal, dtype=np.float32)

    n_freq = len(precomp_list)
    file_features = np.empty((n_freq * 60,), dtype=np.float32)

    for i, (y_final, h, c1, c2, c3, c4, c5, phi_dc, a) in enumerate(precomp_list):
        u_ac_buf = simulate_with_force(y_final, new_len, h, c1, c2, c3, c4, c5, phi_dc, a, signal)
        feats = extract_features(u_ac_buf)
        file_features[i*60:(i+1)*60] = feats

    return file_features


# Ridge functions unchanged (good as-is)
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
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)

    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    
    results = np.array([lam, train_accuracy, accuracy], dtype=np.float64)
    
    return results


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--u_dc', type=float, required=True, help='Value of u_dc to process')
args = parser.parse_args()


if __name__ == '__main__':

    lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])

    train_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    train_filenames = np.loadtxt(train_files_path, dtype=str)
    val_filenames = np.loadtxt(val_files_path, dtype=str)
    filenames = np.concatenate([train_filenames, val_filenames])

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    mu = 1.0
    # u_dc = 0.1
    u_dc = args.u_dc
    ratios = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3])
    f_values = np.linspace(1000, 50000, 100, dtype=int)
    a_crits = np.load(f"/scratch/almo2783/scratch/test/a-crit/a-crits/a-crit-u-dc-{u_dc:.1f}.npy")
    train_accs = []
    val_accs = []
    lambdas = []

    print(f"Processing u_dc = {u_dc:.1f}")

    for ratio in ratios:
        print(f"\nProcessing ratio {ratio}")
        a_values = a_crits * ratio

        # create a tuple of parameters as (f, a, u_dc)
        parameter_tuples = [
            (f_values[i], a_values[i], u_dc)
            for i in range(len(f_values))
        ]

        # Parallel precomputation of transients
        precomp_list = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
            delayed(compute_transient)(params) for params in parameter_tuples
        )

        # Parallel over files (each file handles all frequencies)
        results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
            delayed(process_one_file)(fname, precomp_list) for fname in filenames
        )

        state_matrix = np.vstack(results)

        # Split (adjust indices based on your subsample split)
        train_state = state_matrix[:len(labels_train)]
        val_state = state_matrix[len(train_filenames):]

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(train_state)
        X_val_std = scaler.transform(val_state)

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
        # lambdas   = outputs_arr[:, 0]
        # train_acc = outputs_arr[:, 1] * 100
        # val_acc   = outputs_arr[:, 2] * 100

        lambda_grid = outputs_arr[:, 0]
        train_acc   = outputs_arr[:, 1] * 100
        val_acc     = outputs_arr[:, 2] * 100


        idx_best = np.argmax(val_acc)

        best_val    = val_acc[idx_best]
        best_train  = train_acc[idx_best]
        # best_lambda = lambdas[idx_best]
        best_lambda = lambda_grid[idx_best]


        lambdas.append(best_lambda)
        train_accs.append(best_train)
        val_accs.append(best_val)

        # target = results[2]
        print("\nRidge regression results on validation set:")
        print(f"best Lambda: {best_lambda}")
        print(f"Training acc: {best_train:2f} %")
        print(f"Validation acc: {best_val:.2f} %")

    
    np.save(f"best_lambdas_u_dc-{u_dc:.1f}.npy", np.array(lambdas))
    np.save(f"training_acc_u_dc-{u_dc:.1f}.npy", np.array(train_accs))
    np.save(f"validation_acc_u_dc-{u_dc:.1f}.npy", np.array(val_accs))