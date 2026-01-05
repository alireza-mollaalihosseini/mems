import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed
import warnings
import optuna
import json

# Suppress harmless FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the parameters for 8 sensors: only f, a, u_dc since others are hardcoded
sensor_params = [
    # f, a, u_dc
    (1777,  0.5, 0.4),
    (3555,  0.5, 0.4),
    (5333,  0.5, 0.4),
    (7111,  0.5, 0.4),
    (8888,  0.5, 0.4),
    (10666, 0.5, 0.4),
    (12444, 0.5, 0.4),
    (14222, 0.5, 0.4)
]

# Nominal frequency bands (in bin indices)
nominal_freq_bands = [
    (10, 2666),
    (2666, 4444),
    (4444, 6222),
    (6222, 8000),
    (8000, 9777),
    (9777, 11555),
    (11555, 13333),
    (13333, 16000)
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

def ridge_closed_form(X_train, Y_train, lam=1e-3):
    n_features = X_train.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X_train.T @ X_train + lam * I) @ X_train.T @ Y_train

def evaluate_accuracy(X_train, Y_train, X_test, Y_test, lam=1e-3):
    W = ridge_closed_form(X_train, Y_train, lam)
    Y_pred = X_test @ W
    acc = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_test, axis=1))
    return acc


def build_state_matrix(file_list_path, f, a, u_dc, mu, low, high):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation params
    u_max = 1.0
    omega_0 = 2 * np.pi * f
    alpha = 12.5
    Q_0 = 50.0
    tau = 0.001
    beta = 1066
    gamma = 1.62e7
    R = 12.5
    kappa = 1e6
    h = 1e-6 * omega_0
    T = 50.0 * omega_0

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

    return state_matrix[:, low:high]


def process_sensor(params_tuple, train_file_list_path, val_file_list_path, mu, low, high):
    """
    Builds band-limited state matrices for a sensor.
    params_tuple: (f, a, u_dc)
    """
    f, a, u_dc = params_tuple

    state_train = build_state_matrix(train_file_list_path, f, a, u_dc, mu, low, high)
    state_val   = build_state_matrix(val_file_list_path,   f, a, u_dc, mu, low, high)

    return state_train, state_val


def simulate_all_sensors(params_dict, n_jobs=None):

    mu = 1.0
    train_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    # Build a local copy of sensor_params and replace f, a, u_dc using params_dict
    sensor_params_mod = []
    bands_mod = []
    nominal_lows = [b[0] for b in nominal_freq_bands]
    nominal_highs = [b[1] for b in nominal_freq_bands]
    fft_len = 24001
    for i, tpl in enumerate(sensor_params):
        fi = params_dict.get(f'f_{i}', tpl[0])
        ai = params_dict.get(f'a_{i}', tpl[1])
        u_dci = params_dict.get(f'u_dc_{i}', tpl[2])
        sensor_params_mod.append((float(fi), float(ai), float(u_dci)))

        low_i = int(params_dict.get(f'low_{i}', nominal_lows[i]))
        high_i = int(params_dict.get(f'high_{i}', nominal_highs[i]))
        # Ensure valid range
        low_i = max(0, min(low_i, fft_len - 1))
        high_i = max(low_i + 1, min(high_i, fft_len))
        bands_mod.append((low_i, high_i))

    # Run sensors in parallel
    if n_jobs is None:
        n_jobs = 64
    results = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(
        delayed(process_sensor)(sensor_params_mod[idx], train_file_list_path, val_file_list_path, mu, *bands_mod[idx])
        for idx in range(len(sensor_params_mod))
    )

    # Collect all results
    X_train_all, X_val_all = zip(*results)

    X_train_combined = np.concatenate(X_train_all, axis=1)
    X_val_combined   = np.concatenate(X_val_all, axis=1)
    
    # Standardize using training set stats
    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(X_train_combined)
    state_val_std   = scaler.transform(X_val_combined)

    return state_train_std, state_val_std


def objective(trial):
    params = {}
    params['f_0'] = trial.suggest_float('f_0', 400, 2500)
    params['f_1'] = trial.suggest_float('f_1', 2500, 4500)
    params['f_2'] = trial.suggest_float('f_2', 4500, 6000)
    params['f_3'] = trial.suggest_float('f_3', 6000, 8000)
    params['f_4'] = trial.suggest_float('f_4', 8000, 10000)
    params['f_5'] = trial.suggest_float('f_5', 10000, 12000)
    params['f_6'] = trial.suggest_float('f_6', 12000, 14000)
    params['f_7'] = trial.suggest_float('f_7', 14000, 16000)

    nominal_lows = [10, 2666, 4444, 6222, 8000, 9777, 11555, 13333]
    nominal_highs = [2666, 4444, 6222, 8000, 9777, 11555, 13333, 16000]
    fft_len = 24001
    delta = 500  # adjustment range for bands
    for i in range(8):
        params[f'a_{i}'] = trial.suggest_float(f'a_{i}', -5, 5)
        params[f'u_dc_{i}'] = trial.suggest_float(f'u_dc_{i}', 0.01, 1.0)
        low_min = max(0, nominal_lows[i] - delta)
        low_max = nominal_lows[i] + delta
        params[f'low_{i}'] = trial.suggest_int(f'low_{i}', low_min, low_max)
        high_min = nominal_highs[i] - delta
        high_max = min(fft_len, nominal_highs[i] + delta)
        params[f'high_{i}'] = trial.suggest_int(f'high_{i}', high_min, high_max)

    # simulate all sensors with the sampled parameters
    X_train, X_test = simulate_all_sensors(params)

    Y_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    Y_test   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # evaluate using ridge regression
    accuracy = evaluate_accuracy(X_train, Y_train, X_test, Y_test, lam=1e4)
    
    return accuracy

    
if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="multi_sensor_optimization",
        storage="sqlite:///8_sensor_opt.db",
        load_if_exists=True
    )

    # # Optional: Print current trial count for confirmation
    # print(f"Loaded study with {len(study.trials)} existing trials.")

    study.optimize(objective, n_trials=100, gc_after_trial=True)

    # # Optional: Print updated trial count after new runs
    # print(f"Study now has {len(study.trials)} total trials.")

    # After optimization
    study.trials_dataframe().to_csv("8_sensor_results.csv", index=False)

    print("Best parameters:")
    for key, val in study.best_params.items():
        print(f"{key}: {val:.4f}")
    print(f"Best accuracy: {study.best_value:.4f}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
