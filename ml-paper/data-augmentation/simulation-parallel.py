import sys
import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===========================================
# AUGMENTATION FUNCTIONS
# ===========================================

def augment_audio(signal, sr, rng):
    """Apply random augmentations to audio, keeping length fixed."""
    sig = np.copy(signal)

    # Random gain (amplitude scaling)
    if rng.random() < 0.3:
        gain = rng.uniform(0.8, 1.2)
        sig *= gain

    # Add Gaussian noise
    if rng.random() < 0.3:
        noise = rng.normal(0, 0.005 * np.std(sig), size=sig.shape)
        sig += noise

    # Random frequency-domain filtering (simulate mild coloration)
    if rng.random() < 0.3:
        cutoff = rng.uniform(200, sr / 2 * 0.8)
        b, a = butter(2, cutoff / (sr / 2), btype='low')
        sig = lfilter(b, a, sig)

    # Random polarity flip
    if rng.random() < 0.2:
        sig = -sig

    # Random circular shift (keeps same length)
    if rng.random() < 0.3:
        shift = rng.integers(0, len(sig))
        sig = np.roll(sig, shift)

    sig = np.clip(sig, -1.0, 1.0)
    return sig.astype(np.float32)


# ===========================================
# SIMULATION FUNCTIONS
# ===========================================

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
    y = np.zeros(4)
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


# ===========================================
# MAIN PROCESSING PIPELINE
# ===========================================

def process_file(fname, idxs, b_filt, a_filt, y_final, N_force, h,
                 c1, c2, c3, c4, c5, phi_dc, a, fft_len, seed, use_augmentation=False):
    rng = np.random.default_rng(seed + hash(fname) % 10000)
    data, sr = sf.read(fname)
    signal = data[idxs]

    if use_augmentation:
        signal = augment_audio(signal, sr, rng)

    signal = lfilter(b_filt, a_filt, signal)
    signal_buf = np.zeros(N_force, dtype=np.float64)
    signal_buf[:len(signal)] = signal[:N_force]

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal_buf)
    fft_vals = np.fft.rfft(u_ac_buf)
    return np.abs(fft_vals[:fft_len]).astype(np.float32)


def build_state_matrix(file_list_path, a, u_dc, mu, seed=42, use_augmentation=False, n_augment_per_file=1):
    filenames = np.loadtxt(file_list_path, dtype=str)
    n_files = len(filenames)

    # Simulation constants
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
    u_max = 1.0
    h = 1e-6 * omega_0
    T = 50.0 * omega_0

    N_trans = int(T / h)
    N_force = int((1.0 * omega_0) / h)
    l_0 = (alpha * gamma * u_max ** 2) / (beta * R ** 2 * omega_0 ** 2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / (u_max)
    c5 = mu / (l_0 * omega_0 ** 2)
    phi_dc = u_dc / u_max

    data0, sr = sf.read(filenames[0])
    new_sr = int((1.0 * omega_0) / h)
    frac = new_sr / sr
    idxs = (np.arange(int(len(data0) * frac)) / frac).astype(np.int64)
    b_filt, a_filt = butter(4, (0.49 * sr), fs=sr, btype='low')

    fft_len = 24001
    y_final = simulate_transient(N_trans, h, c1, c2, c3, c4, phi_dc, a)

    results = Parallel(n_jobs=64, backend='threading')(
        delayed(process_file)(
            fname, idxs, b_filt, a_filt, y_final, N_force, h,
            c1, c2, c3, c4, c5, phi_dc, a, fft_len, seed, use_augmentation
        )
        for fname in filenames
        for _ in range(n_augment_per_file)
    )

    return np.vstack(results), filenames.repeat(n_augment_per_file)


# ===========================================
# RIDGE REGRESSION + MAIN
# ===========================================

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


def main(train_file_list_path, val_file_list_path, test_file_list_path, a, u_dc, mu):
    lam = 1e4
    seed = 42
    use_augmentation = True
    n_augment_per_file = 5

    state_train, train_files_expanded = build_state_matrix(train_file_list_path, a, u_dc, mu, seed, use_augmentation, n_augment_per_file)
    state_val, _   = build_state_matrix(val_file_list_path, a, u_dc, mu, seed, use_augmentation=False, n_augment_per_file=1)
    state_test, _  = build_state_matrix(test_file_list_path, a, u_dc, mu, seed, use_augmentation=False, n_augment_per_file=1)

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_test  = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_test.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Repeat labels for augmented files
    labels_train = np.repeat(labels_train, n_augment_per_file, axis=0)
    # labels_test  = np.repeat(labels_test, n_augment_per_file=1, axis=0)
    # labels_val   = np.repeat(labels_val, n_augment_per_file=1, axis=0)

    scaler = StandardScaler()
    state_train_std = scaler.fit_transform(state_train)
    state_test_std  = scaler.transform(state_test)
    state_val_std   = scaler.transform(state_val)

    results_test, cm_test = ridge_regression_fast(state_train_std, labels_train, state_test_std, labels_test, lam, a, u_dc)
    results_val, cm_val   = ridge_regression_fast(state_train_std, labels_train, state_val_std, labels_val, lam, a, u_dc)
                                                                              
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/results_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt", results_test.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/conf_matrix_test-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt", cm_test, fmt="%.5f")

    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/results_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt", results_val.reshape(1, -1), fmt="%.5f")
    np.savetxt(f"/scratch/almo2783/scratch/ml-paper/data-augmentation/results/conf_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt", cm_val, fmt="%.5f")


if __name__ == '__main__':
    a, u_dc, mu = (0.44, 0.4, 1.0)

    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    main(train_files, val_files, test_files, a, u_dc, mu)
