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
from scipy.stats import skew, kurtosis


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


def process_file(fname, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a):
    data, sr = sf.read(fname)
    new_sr = int(1e6)
    frac = new_sr / sr
    idxs = (np.arange(int(len(data)*frac)) / frac).astype(np.int64)
    signal = data[idxs]

    # signal_buf = np.zeros(N_force, dtype=np.float64)
    # signal_buf[:len(signal)] = signal

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal)

    du  = np.diff(u_ac_buf)
    d2u = np.diff(u_ac_buf, n=2)

    feats = [
        np.mean(u_ac_buf),
        np.std(u_ac_buf),
        np.mean(du),
        np.std(du),
        np.mean(d2u),
        np.std(d2u),

        np.max(u_ac_buf),
        np.argmax(u_ac_buf),
        np.max(du),
        np.argmax(du),
        np.max(d2u),
        np.argmax(d2u),

        np.sqrt(np.mean(u_ac_buf**2)),  # RMS
        np.ptp(u_ac_buf),
        skew(u_ac_buf),
        kurtosis(u_ac_buf)
    ]

    return np.array(feats)


def build_state_matrix(train_file_list_path, val_file_list_path, results_dir, a, u_dc, mu, f):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)

    n_train = len(train_filenames)
    n_val   = len(val_filenames)

    # Combine for single parallel processing
    filenames = np.concatenate([train_filenames, val_filenames])
    n_files = len(filenames)

    # Simulation params
    alpha, omega_0, Q_0, tau, beta, gamma, R, kappa = 19.2, f*2*np.pi, 50.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
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

    # Initial transient
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(process_file)(fname, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a)
        for fname in filenames
    )

    state_matrix = np.vstack(results)

    # Save result
    output_path = results_dir + f"/f-{int(f)}.npz"
    np.savez_compressed(output_path, state_matrix)



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--f', type=float, required=True, help='Value of f to process')
args = parser.parse_args()

if __name__ == '__main__':

    a = 0.44
    mu = 1.0 
    u_dc = 0.4
    f = args.f

    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/multi-sens/test/results"
    os.makedirs(results_dir, exist_ok=True)

    build_state_matrix(train_files, val_files, results_dir, a, u_dc, mu, f)