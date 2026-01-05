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

# f_values = np.array([445, 490, 582, 591, 800, 900, 1000, 1200, 1600, 2400, 3000, 3500, 4000, 5600, 6000, 7000, 8000, 9000, 12000, 14000, 15000])
# band_widths = np.array([50, 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50  , 100 , 200 , 400 , 400 , 400 , 400 , 800 , 700,  700,  1000, 1500,  1500,  1500])
# band_widths = np.array([50, 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50  , 100 , 200 , 200 , 200 , 200 , 200 , 400 , 400,  400,  500, 500,  500,  500])
# band_widths = np.array([500, 200, 300, 200, 50 , 50 , 50  , 100 , 200 , 200 , 200 , 200 , 200, 200, 400 , 400,  400,  500, 500,  500,  500])          

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
    (195.0, 695.0),
    (390.0, 590.0),
    (432.0, 732.0),
    (491.0, 691.0),
    (775.0, 825.0),
    (875.0, 925.0),
    (975.0, 1025.0),
    (1150.0, 1250.0),
    (1500.0, 1700.0),
    (2300.0, 2500.0),
    (2900.0, 3100.0),
    (3400.0, 3600.0),
    (3900.0, 4100.0),
    (5500.0, 5700.0),
    (5800.0, 6200.0),
    (6800.0, 7200.0),
    (7800.0, 8200.0),
    (8750.0, 9250.0),
    (11750.0, 12250.0),
    (13750.0, 14250.0),
    (14750.0, 15250.0)
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


def build_state_matrix(file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation params
    u_max = 1.0
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

    return state_matrix

# def process_sensor(idx, params, train_file_list_path, val_file_list_path, mu):
#     """Runs one sensor simulation (or special 7th sensor case)."""
#     (alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc) = params

#     print(f"Simulating Sensor {idx+1} with ω₀={omega_0:.2f} Hz")

#     state_train = build_state_matrix(train_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)
#     state_val   = build_state_matrix(val_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)
    
#     state_train = state_train[:, freq_bands[idx][0]:freq_bands[idx][1]]
#     state_val   = state_val[:, freq_bands[idx][0]:freq_bands[idx][1]]
    
#     return state_train, state_val


def process_sensor(idx, params_tuple_or_dict, train_file_list_path, val_file_list_path, mu):
    """
    Builds band-limited state matrices for sensor idx.
    params_tuple_or_dict can be a tuple (alpha,...,u_dc) or a dict with 'a_<i>' and 'u_dc_<i>' already substituted.
    """
    # get sensor params
    if isinstance(params_tuple_or_dict, tuple) or isinstance(params_tuple_or_dict, list):
        params = params_tuple_or_dict
    elif isinstance(params_tuple_or_dict, dict):
        # this branch is only used if simulate_all_sensors passes sensor-specific tuple
        raise ValueError("process_sensor expects sensor tuple; modify simulate_all_sensors to pass tuples.")
    else:
        raise ValueError("Unsupported params type for process_sensor")

    (alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc) = params

    print(f"Simulating Sensor {idx+1} with ω₀={omega_0:.2f} Hz, a={a:.4f}, u_dc={u_dc:.4f}")

    # band = freq_bands[idx]
    state_train = build_state_matrix(train_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)
    state_val   = build_state_matrix(val_file_list_path,   alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu)

    state_train = state_train[:, freq_bands[idx][0]:freq_bands[idx][1]]
    state_val   = state_val[:, freq_bands[idx][0]:freq_bands[idx][1]]

    return state_train, state_val


def simulate_all_sensors(params_dict, n_jobs=None):

    mu = 1.0
    train_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    # replace a and u_dc values from params into sensor_params
    # Build a local copy of sensor_params and replace a and u_dc using params_dict
    sensor_params_mod = []
    for i, tpl in enumerate(sensor_params):
        tpl = list(tpl)
        ai = params_dict.get(f'a_{i}', tpl[9])         # default to original if missing
        u_dci = params_dict.get(f'u_dc_{i}', tpl[10])
        tpl[9] = float(ai)
        tpl[10] = float(u_dci)
        sensor_params_mod.append(tuple(tpl))

    # # choose sensible n_jobs
    # max_workers = os.cpu_count() or 1
    # if n_jobs is None:
    #     n_jobs = min(64, max_workers, len(sensor_params_mod))
    n_jobs = 64

    # Run sensors in parallel
    results = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(
        delayed(process_sensor)(idx, sensor_params_mod[idx], train_file_list_path, val_file_list_path, mu)
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
    for i in range(7):
        params[f'a_{i}'] = trial.suggest_float(f'a_{i}', -2, 2)
        params[f'u_dc_{i}'] = trial.suggest_float(f'u_dc_{i}', 0.01, 1.0)
    
    # simulate all sensors with the sampled parameters
    X_train, X_test = simulate_all_sensors(params)

    Y_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    Y_test   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # evaluate using ridge regression
    accuracy = evaluate_accuracy(X_train, Y_train, X_test, Y_test, lam=1e4)
    
    return accuracy

    
if __name__ == "__main__":

    # study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), study_name="feedback_optimization")
    # study.optimize(objective, n_trials=100, n_jobs=64)

    study = optuna.create_study(
        direction="maximize",
        study_name="multi_sensor_a_udc_optimization",
        storage="sqlite:///multi_sensor_opt.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100, gc_after_trial=True)

    # After optimization
    study.trials_dataframe().to_csv("multi_sensor_results.csv", index=False)

    print("Best parameters:")
    for key, val in study.best_params.items():
        print(f"{key}: {val:.4f}")
    print(f"Best accuracy: {study.best_value:.4f}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
