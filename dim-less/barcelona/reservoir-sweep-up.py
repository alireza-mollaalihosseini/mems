import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
# from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
def simulate_transient(N, h, c1, c2, c3, c4, phi_dc, a, y0):

  y  = y0
  # y[0] = 1e-9
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


def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    return np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ Y_train)

def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam, a, u_dc):
    # Add bias term
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    # Train ridge regression
    W = ridge_closed_form(X_train_b, Y_train, lam)

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

def process_file(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, fft_len):
    data, _ = sf.read(fname)
    signal = data[idxs]
    signal = lfilter(b_filt, a_filt, signal)

    signal_buf = np.zeros(N_force, dtype=np.float64)
    signal_buf[:len(signal)] = signal

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal_buf)
    fft_vals = np.fft.rfft(u_ac_buf)
    return np.abs(fft_vals[:fft_len]).astype(np.float32)


def build_state_matrix(file_list_path, a, u_dc, mu, y_0):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation params
    omega_0, Q_0 = 53956.46373431294, 50.0
    alpha, beta = 19.2, 1066.0
    gamma, R = 1.62e7, 12.5
    tau = 0.001
    kappa = 0.602e6
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
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a, y_0)

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

    # results = Parallel(n_jobs=-1)(
    #     delayed(process_file)(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, fft_len)
    #     for fname in filenames
    # )

    # state_matrix = np.vstack(results)

    return y_final, state_matrix
    

# -------------------------
# Main
# -------------------------
def main(train_file_list_path, val_file_list_path, test_file_list_path, a_values, u_dc, mu):

    lam = 1e4
    kick = 2e-3
    y_0 = np.zeros(4)
    y_0[0] = 1e-9

    for a in a_values:

      y_0, state_train = build_state_matrix(train_file_list_path, a, u_dc, mu, y_0)
      y_0, state_val   = build_state_matrix(val_file_list_path, a, u_dc, mu, y_0)
      y_0, state_test  = build_state_matrix(test_file_list_path, a, u_dc, mu, y_0)

      labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
      labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
      labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

      # Standardize using training set stats
      scaler = StandardScaler()
      state_train_std = scaler.fit_transform(state_train)
      state_test_std  = scaler.transform(state_test)
      state_val_std   = scaler.transform(state_val)

      # Train on train set, test on test set
      results_test, cm_test = ridge_regression_fast(
          state_train_std, labels_train, state_test_std, labels_test,
          lam, a, u_dc
      )

      # Train on train set, eval on val set
      results_val, cm_val = ridge_regression_fast(
          state_train_std, labels_train, state_val_std, labels_val,
          lam, a, u_dc
      )

      # Save
      np.savetxt(f"/scratch/almo2783/scratch/dim-less/barcelona/results_up/results_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
                results_test.reshape(1, -1), fmt="%.5f")
      np.savetxt(f"/scratch/almo2783/scratch/dim-less/barcelona/results_up/conf_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
                cm_test, fmt="%.5f")

      np.savetxt(f"/scratch/almo2783/scratch/dim-less/barcelona/results_up/results_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
                results_val.reshape(1, -1), fmt="%.5f")
      np.savetxt(f"/scratch/almo2783/scratch/dim-less/barcelona/results_up/conf_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt",
                cm_val, fmt="%.5f")

      # add kick
      y_0[0] += kick


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # a = float(sys.argv[1])
        u_dc = float(sys.argv[1])
        mu = float(sys.argv[2])
    else:
        # a = 0.5
        u_dc = 0.7
        mu = 1.0

    a_values = np.linspace(-2, 4, 2)

    main(
        '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv',
        '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv',
        '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv',
        a_values,
        u_dc,
        mu
    )