import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed

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


def process_file(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a):
    data, _ = sf.read(fname)
    signal = data[idxs]
    signal = lfilter(b_filt, a_filt, signal)

    signal_buf = np.zeros(N_force, dtype=np.float64)
    signal_buf[:len(signal)] = signal

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal_buf)
    
    return u_ac_buf.astype(np.float32)


def build_state_matrix(file_list_path, output_path, a, u_dc, mu):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation params
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
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

    # Initial transient
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    results = Parallel(n_jobs=64, backend="threading", verbose=1)(
        delayed(process_file)(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a)
        for fname in filenames
    )

    state_matrix = np.vstack(results)

    # Save result
    np.savez_compressed(output_path, state_matrix)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        a = float(sys.argv[1])
        u_dc = float(sys.argv[2])
        mu = float(sys.argv[3])
    else:
        # best val point
        a = 0.44
        u_dc = 0.4
        mu = 1.0

    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    build_state_matrix(
        train_files,
        f'/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_train-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu
    )

    build_state_matrix(
        val_files,
        f'/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu
    )

    build_state_matrix(
        test_files,
        f'/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/state_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}.npz',
        a, u_dc, mu
    )