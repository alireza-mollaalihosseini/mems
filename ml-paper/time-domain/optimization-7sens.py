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
from scipy.signal import decimate

# Suppress harmless FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the parameters for 7 sensors
sensor_params = [
    # alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc
    (12.5, 445.0, 2796.0174616949157, 29.0, 0.0034, 294.11764705882354, 2256560.0, 13.4, 147800.0, 0.16, 0.1, 5000),
    (12.5, 490.0, 3078.7608005179973, 33.0, 0.0039, 256.4102564102564, 1800000.0, 13.4, 162540.0, 0.04, 0.9, 5000),
    (12.5, 582.0, 3656.813848778519, 50.0, 0.0044, 227.27272727272725, 1356000.0, 13.4, 195030.0, 0.02, 0.9, 5000),
    (12.5, 591.0, 3713.3625165431354, 45.0, 0.0039, 256.4102564102564, 1800000.0, 13.4, 214190.0, 0.6, 0.1, 5000),
    (12.5, 1109.0, 6968.052505662161, 57.5, 0.0051, 196.07843137254903, 2669280.0, 12.92, 400980.0, 0.06, 1.0, 5000),
    (12.5, 1161.0, 7294.778141635499, 57.0, 0.0057, 175.43859649122805, 2391680.0, 13.23, 361630.0, 0.44, 0.1, 5000),
    (12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0, 0.44, 0.4, 5000)
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
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_test_b  = np.hstack((X_test,  np.ones((X_test.shape[0], 1), dtype=X_test.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)
    Y_pred = X_test_b @ W
    acc = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_test, axis=1))
    return acc



def build_state_matrix(file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu, new_fs):
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

    # # Preallocate state_matrix
    # state_matrix = np.zeros((n_files, new_fs+1), dtype=np.float32)

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
        # fft_vals = np.fft.rfft(u_ac_buf)
        # Downsampling timeseries reservoir response (u_ac)
        original_fs = 1e6  # 1 MHz
        downsample_factor = int(original_fs / new_fs)  # 200

        # Downsample: Applies FIR low-pass filter (cutoff at new Nyquist = new_fs / 2) by default
        downsampled = decimate(u_ac_buf, downsample_factor, ftype='fir')
        if i == 0:
            state_matrix = np.zeros((n_files, len(downsampled)), dtype=np.float32)
        state_matrix[i] = downsampled.astype(np.float32)

    return state_matrix



def process_sensor(idx, params_tuple_or_dict, train_file_list_path, val_file_list_path, mu):

    # get sensor params
    if isinstance(params_tuple_or_dict, tuple) or isinstance(params_tuple_or_dict, list):
        params = params_tuple_or_dict
    elif isinstance(params_tuple_or_dict, dict):
        # this branch is only used if simulate_all_sensors passes sensor-specific tuple
        raise ValueError("process_sensor expects sensor tuple; modify simulate_all_sensors to pass tuples.")
    else:
        raise ValueError("Unsupported params type for process_sensor")

    (alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, new_fs) = params

    state_train = build_state_matrix(train_file_list_path, alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu, new_fs)
    state_val   = build_state_matrix(val_file_list_path,   alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa, a, u_dc, mu, new_fs)

    return state_train, state_val



def simulate_all_sensors(params_dict, mu, n_jobs=None):

    train_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_file_list_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    # replace a and u_dc values from params into sensor_params
    # Build a local copy of sensor_params and replace a and u_dc using params_dict
    sensor_params_mod = []
    for i, tpl in enumerate(sensor_params):
        tpl = list(tpl)
        ai = params_dict.get(f'a_{i}', tpl[9])
        u_dci = params_dict.get(f'u_dc_{i}', tpl[10])
        new_fsi = params_dict.get(f'new_fs_{i}', tpl[11])
        tpl[9] = float(ai)
        tpl[10] = float(u_dci)
        tpl[11] = int(new_fsi)

        sensor_params_mod.append(tuple(tpl))

    # Run sensors in parallel
    results = Parallel(n_jobs=64, backend='threading', verbose=5)(
        delayed(process_sensor)(idx, sensor_params_mod[idx], train_file_list_path, val_file_list_path, mu)
        for idx in range(len(sensor_params_mod))
    )

    # Collect all results
    X_train_all, X_val_all = zip(*results)

    X_train_combined = np.concatenate(X_train_all, axis=1)
    X_val_combined   = np.concatenate(X_val_all, axis=1)

    return X_train_combined, X_val_combined


def objective(trial):

    params = {}
    for i in range(7):
        params[f'a_{i}'] = trial.suggest_float(f'a_{i}', -4, 4)
        params[f'u_dc_{i}'] = trial.suggest_float(f'u_dc_{i}', 0, 1.0)
        params[f'new_fs_{i}'] = trial.suggest_int(f'new_fs_{i}', 1, 5000)

    mu = trial.suggest_float('mu', 0, 5)
    lam = trial.suggest_float('lambda', 1e-5, 1e6, log=True)

    # simulate all sensors with the sampled parameters
    X_train_all, X_val_all = simulate_all_sensors(params, mu)
    
    # Standardize using training set stats
    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(X_train_all)
    state_val_std   = scaler.transform(X_val_all)

    Y_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    Y_test   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # evaluate using ridge regression
    accuracy = evaluate_accuracy(state_train_std, Y_train, state_val_std, Y_test, lam=lam)
    
    return accuracy

    
if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="7sens_timesample_optimization",
        storage="sqlite:///7sens_time_opt.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100, gc_after_trial=True)