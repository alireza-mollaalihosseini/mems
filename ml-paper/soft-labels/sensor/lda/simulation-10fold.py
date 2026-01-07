import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pywt
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


def process_file(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, wavelet="db4", maxlevel=5, mode="symmetric"):
    data, _ = sf.read(fname)
    signal = data[idxs]
    signal = lfilter(b_filt, a_filt, signal)

    signal_buf = np.zeros(N_force, dtype=np.float64)
    signal_buf[:len(signal)] = signal

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal_buf)
    data = u_ac_buf.astype(np.float32)
    data -= np.mean(data)
    data /= (np.max(np.abs(data)) + 1e-12)

    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order="freq")

    features = []
    total_energy = 0.0
    energies = []
    
    # -------------------------
    # First pass: energies
    # -------------------------
    for node in nodes:
        c = node.data
        e = np.sum(c**2)
        energies.append(e)
        total_energy += e

    energies = np.array(energies) + 1e-16

    # -------------------------
    # Second pass: rich features
    # -------------------------
    for i, node in enumerate(nodes):
        c = node.data
        abs_c = np.abs(c)

        # Energy features
        log_energy = np.log10(energies[i])
        rel_energy = energies[i] / total_energy
        rms = np.sqrt(np.mean(c**2))

        # Distribution
        mean_abs = np.mean(abs_c)
        std = np.std(c)
        skewness = skew(c)
        kurt = kurtosis(c)

        # Sparsity / entropy
        p = abs_c / (np.sum(abs_c) + 1e-16)
        shannon_entropy = -np.sum(p * np.log2(p + 1e-16))
        l1_l2 = np.sum(abs_c) / (np.sqrt(np.sum(c**2)) + 1e-16)

        # Temporal structure
        zcr = np.mean(np.diff(np.sign(c)) != 0)
        crest = np.max(abs_c) / (rms + 1e-16)

        features.extend([
            log_energy,
            rel_energy,
            rms,
            mean_abs,
            std,
            skewness,
            kurt,
            shannon_entropy,
            l1_l2,
            zcr,
            crest
        ])

    return np.array(features, dtype=np.float32)


def build_state_matrix(train_file_list_path, val_file_list_path, test_file_list_path, a, u_dc, mu):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    filenames = np.concatenate([train_filenames, val_filenames, test_filenames])
    n_files = len(filenames)

    # Simulation params
    # alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 19.2, 8587.437915074492, 53956.46373431294, 50.0, 0.001, 1066.0, 1.62e7, 12.5, 0.602e6
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

    results = Parallel(n_jobs=64, verbose=1, backend="multiprocessing")(
        delayed(process_file)(fname, idxs, b_filt, a_filt, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a)
        for fname in filenames
    )

    state_matrix = np.vstack(results)

    return state_matrix


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--a', type=float, required=True, help='Value of a to process')
# args = parser.parse_args()

if __name__ == '__main__':

    a = 0.44
    u_dc = 0.5
    mu = 1.0
    
    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'

    # Load labels
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
    results_dir = f"/scratch/almo2783/scratch/ml-paper/soft-labels/sensor/lda"
    os.makedirs(results_dir, exist_ok=True)

    state_matrix = build_state_matrix(train_files, val_files, test_files, a, u_dc, mu)

    # -----------------------------
    # Bootstrapping hyperparameters
    # -----------------------------
    bootstrap_iters = 5          # Number of bootstrapping iterations (3–10 is typical)
    beta = 0.95                  # Weight on original (noisy) labels; higher = more conservative

    # -----------------------------
    # K-Fold Training Loop
    # -----------------------------
    for fold, (train_idx, test_idx) in enumerate(kf.split(state_matrix), start=1):
        print(f"\n========== Fold {fold}/{n_splits} ==========")

        X_train = state_matrix[train_idx]
        X_test = state_matrix[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Ensure y_train/y_test are one-hot (required for bootstrapping)
        if y_train.ndim == 1:
            n_classes = len(np.unique(y_train))
            y_train = np.eye(n_classes)[y_train]
            y_test = np.eye(n_classes)[y_test]
        elif y_train.shape[1] == len(np.unique(np.argmax(y_train, axis=1))):
            pass  # already one-hot
        else:
            raise ValueError("Unexpected label format")
        
        # Scale per fold
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # Bootstrapping to refine (soften) training labels
        current_soft_labels = y_train.astype(np.float64).copy()
        
        for it in range(bootstrap_iters):
            current_hard_labels = np.argmax(current_soft_labels, axis=1)
            
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X_train_std, current_hard_labels)
            
            train_proba = lda.predict_proba(X_train_std)
            current_soft_labels = beta * y_train + (1 - beta) * train_proba
        
        # Final model on refined hard labels
        final_hard_labels = np.argmax(current_soft_labels, axis=1)
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train_std, final_hard_labels)
        
        # Predictions and metrics (using original labels for evaluation)
        y_train_true = np.argmax(y_train, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        
        train_pred = lda.predict(X_train_std)
        train_accuracy = accuracy_score(y_train_true, train_pred)
        
        test_pred = lda.predict(X_test_std)
        accuracy = accuracy_score(y_test_true, test_pred)
        precision = precision_score(y_test_true, test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_true, test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_true, test_pred, average='macro', zero_division=0)
        
        results = np.array([train_accuracy, accuracy, precision, recall, f1], dtype=np.float64)
        
        # Save
        np.savetxt(
            f"{results_dir}/fold_results-fold-{fold}.txt",
            results.reshape(1, -1),
            fmt="%.6f"
        )
    
    print("\n✅ K-Fold cross-validation with bootstrapping finished successfully.")