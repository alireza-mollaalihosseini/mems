import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed
import warnings

# Suppress harmless FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the parameters for 21 sensors
sensor_params = [
    # alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc
    (12.5, 445.0, 2796.0174616949157, 29.0, 0.0034, 294.11764705882354, 2256560.0, 13.4, 147800.0, 0.16, 0.1),
    (12.5, 490.0, 3078.7608005179973, 33.0, 0.0039, 256.4102564102564, 1800000.0, 13.4, 162540.0, 0.04, 0.9),
    (12.5, 582.0, 3656.813848778519, 50.0, 0.0044, 227.27272727272725, 1356000.0, 13.4, 195030.0, 0.02, 0.9),
    (12.5, 591.0, 3713.3625165431354, 45.0, 0.0039, 256.4102564102564, 1800000.0, 13.4, 214190.0, 0.6, 0.1),
    (12.5, 800,   2 * np.pi * 800, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 900,   2 * np.pi * 900, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 1000,  2 * np.pi * 1000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 1200,  2 * np.pi * 1200, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 1600,  2 * np.pi * 1600, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 2400,  2 * np.pi * 2400, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 3000,  2 * np.pi * 3000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 3500,  2 * np.pi * 3500, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 4000,  2 * np.pi * 4000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 5600,  2 * np.pi * 5600, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 6000,  2 * np.pi * 6000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 7000,  2 * np.pi * 7000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 8000,  2 * np.pi * 8000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 9000,  2 * np.pi * 9000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 12000, 2 * np.pi * 12000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 14000, 2 * np.pi * 14000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
    (12.5, 15000, 2 * np.pi * 15000, 50, 0.001, 1066, 1.62e7, 12.5, 1e6, 0.5, 0.4),
]

# Define frequency bands to extract for each sensor
freq_bands = [
    (195, 695),
    (390, 590),
    (432, 732),
    (491, 691),
    (775, 825),
    (875, 925),
    (975, 1025),
    (1150, 1250),
    (1500, 1700),
    (2300, 2500),
    (2900, 3100),
    (3400, 3600),
    (3900, 4100),
    (5500, 5700),
    (5800, 6200),
    (6800, 7200),
    (7800, 8200),
    (8750, 9250),
    (11750, 12250),
    (13750, 14250),
    (14750, 15250)
]

@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
    # k1
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    # k2
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    # k3
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    # k4
    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    # update y
    for i in range(4):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force(y, h, c1, c2, c3, c4, c5, phi_dc, a, k1, k2, k3, k4, y_temp, f_x):
    # k1
    exsi, etta, psy, phi_ac = y
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k1[0] = etta
    k1[1] = -c2 * etta - exsi + psy + c5 * f_x
    k1[2] = -c1 * psy + c1 * min_term
    k1[3] = -c3 * phi_ac + c4 * etta

    # k2
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k2[0] = etta
    k2[1] = -c2 * etta - exsi + psy + c5 * f_x
    k2[2] = -c1 * psy + c1 * min_term
    k2[3] = -c3 * phi_ac + c4 * etta

    # k3
    for i in range(4):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k3[0] = etta
    k3[1] = -c2 * etta - exsi + psy + c5 * f_x
    k3[2] = -c1 * psy + c1 * min_term
    k3[3] = -c3 * phi_ac + c4 * etta

    # k4
    for i in range(4):
        y_temp[i] = y[i] + h * k3[i]
    exsi, etta, psy, phi_ac = y_temp
    temp = a * phi_ac + phi_dc
    min_term = min(temp * temp, 1.0)
    k4[0] = etta
    k4[1] = -c2 * etta - exsi + psy + c5 * f_x
    k4[2] = -c1 * psy + c1 * min_term
    k4[3] = -c3 * phi_ac + c4 * etta

    # update y
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


def ridge_closed_form(X_train, Y_train, lam, a, u_dc):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)

def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Train ridge regression
    W = ridge_closed_form(X_train_b, Y_train, lam, a, u_dc)

    # Predictions
    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    y_train_true = np.argmax(Y_train, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)
    y_eval_true = np.argmax(Y_eval, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    precision = precision_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    recall    = recall_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)
    f1        = f1_score(y_eval_true, y_eval_hats, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(y_eval_true, y_eval_hats)

    results = np.array([a, u_dc, lam, train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
    return results, conf_matrix


def freq_band_extract(state_matrix, band):
    start, end = band
    return state_matrix[:, start:end]


def build_state_matrix(file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation params
    u_max = 1.0
    h = 1e-6 * omega_0
    T = 5.0 * omega_0

    # Derived constants
    N_trans = int(T / h)
    N_force = int((1.0 * omega_0) / h)
    l_0          = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1           = beta / omega_0
    c2           = 1/ Q_0
    c3           = 1 / (tau * omega_0)
    c4           = (kappa * l_0) / (u_max)
    c5           = mu / (l_0 * omega_0**2)
    phi_dc       = u_dc / u_max

    # Precompute interpolation indices & filter
    data0, sr = sf.read(filenames[0])
    new_sr = int((1.0 * omega_0) / h)
    frac = new_sr / sr
    idxs = (np.arange(int(len(data0)*frac)) / frac).astype(np.int64)
    b_filt, a_filt = butter(4, (0.49 * sr), fs=sr, btype='low')

    # Preallocate state_matrix
    fft_len = 24001
    state_matrix = np.zeros((n_files, fft_len), dtype=np.float32)

    # Initial transient
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    # Workspace buffers (reused per file)
    u_ac_buf = np.empty(N_force, dtype=np.float64)
    signal_buf = np.empty(N_force, dtype=np.float64)

    # Process each file independently
    for i, fname in enumerate(filenames):
        data, _ = sf.read(fname)
        signal = data[idxs]
        signal = lfilter(b_filt, a_filt, signal)
        signal_buf[:len(signal)] = signal  # store into fixed buffer
        y0 = y_final.copy()
        u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal_buf)
        fft_vals = np.fft.rfft(u_ac_buf)
        state_matrix[i] = np.abs(fft_vals[:fft_len]).astype(np.float32)

    return state_matrix


def process_sensor(idx, params, train_file_list_path, val_file_list_path, test_file_list_path, mu):
    """Runs one sensor simulation (or special 7th sensor case)."""
    (alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc) = params

    # print(f"Simulating Sensor {idx+1} with ω₀={omega_0:.2f} Hz")

    state_train = build_state_matrix(train_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)
    state_val   = build_state_matrix(val_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)
    state_test  = build_state_matrix(test_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)

    state_train = state_train[:, freq_bands[idx][0]:freq_bands[idx][1]]
    state_val   = state_val[:, freq_bands[idx][0]:freq_bands[idx][1]]
    state_test   = state_test[:, freq_bands[idx][0]:freq_bands[idx][1]]

    # state_train = freq_band_extract(state_train, freq_bands[idx])
    # state_val   = freq_band_extract(state_val, freq_bands[idx])
    # state_test  = freq_band_extract(state_test, freq_bands[idx])

    return state_train, state_val, state_test


def main_all_sensors(train_file_list_path, val_file_list_path, test_file_list_path, a, mu):
    """Parallel version of 7-sensor simulation."""
    lam = 1e4
    u_dc_special = 0.4

    # Run sensors in parallel using 8 cores (1 per sensor)
    results = Parallel(n_jobs=64, backend='threading', verbose=5)(
        delayed(process_sensor)(idx, params, train_file_list_path, val_file_list_path, test_file_list_path, mu)
        for idx, params in enumerate(sensor_params)
    )

    # Collect all results
    X_train_all, X_val_all, X_test_all = zip(*results)

    X_train_combined = np.concatenate(X_train_all, axis=1)
    X_val_combined   = np.concatenate(X_val_all, axis=1)
    X_test_combined  = np.concatenate(X_test_all, axis=1)

    # Standardize using training set stats
    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(X_train_combined)
    state_test_std  = scaler.transform(X_test_combined)
    state_val_std   = scaler.transform(X_val_combined)

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Train on train set, test on test set
    results_test, cm_test = ridge_regression_fast(
        state_train_std, labels_train, state_test_std, labels_test,
        lam, a, u_dc_special
    )

    # Train on train set, eval on val set
    results_val, cm_val = ridge_regression_fast(
        state_train_std, labels_train, state_val_std, labels_val,
        lam, a, u_dc_special
    )

    # Save
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/no-equidist/results/results_test-a-{a}-u_dc-{u_dc_special}-mu-{mu}-lam-{lam:.0e}.txt",
            results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/no-equidist/results/conf_matrix_test-a-{a}-u_dc-{u_dc_special}-mu-{mu}-lam-{lam:.0e}.txt",
            cm_test, fmt="%.5f")

    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/no-equidist/results/results_val-a-{a}-u_dc-{u_dc_special}-mu-{mu}-lam-{lam:.0e}.txt",
            results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/no-equidist/results/conf_matrix_val-a-{a}-u_dc-{u_dc_special}-mu-{mu}-lam-{lam:.0e}.txt",
            cm_val, fmt="%.5f")

    return X_train_combined

    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        a_override = float(sys.argv[1])
        sensor_params[-1] = (*sensor_params[-1][:-2], a_override, sensor_params[-1][-1])
        print(f"Overriding a for last sensor -> {a_override}")
    mu = 1.0
    main_all_sensors(
        '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv',
        '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv',
        '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv',
        a_override,
        mu
    )
