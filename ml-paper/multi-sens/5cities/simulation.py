import os
import numpy as np
import soundfile as sf
from numba import njit
from joblib import Parallel, delayed
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


def features_extraction(signal):
    u_ac_buf = signal.copy()
    du  = np.diff(u_ac_buf)
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

    # Clearance factor – precompute for consistency
    sqrt_abs_mean_u = np.mean(np.sqrt(abs_u))
    sqrt_abs_mean_du = np.mean(np.sqrt(abs_du))
    sqrt_abs_mean_d2u = np.mean(np.sqrt(abs_d2u))

    clearance_u = peak_u / (sqrt_abs_mean_u ** 2) if sqrt_abs_mean_u > 0 else 0.0
    clearance_du = peak_du / (sqrt_abs_mean_du ** 2) if sqrt_abs_mean_du > 0 else 0.0
    clearance_d2u = peak_d2u / (sqrt_abs_mean_d2u ** 2) if sqrt_abs_mean_d2u > 0 else 0.0

    # Skewness & kurtosis on all signals
    skew_u = skew(u_ac_buf)
    kurt_u = kurtosis(u_ac_buf)
    skew_du = skew(du)
    kurt_du = kurtosis(du)
    skew_d2u = skew(d2u)
    kurt_d2u = kurtosis(d2u)

    # Robust statistics
    median_u = np.median(u_ac_buf)
    p25_u, p75_u = np.percentile(u_ac_buf, [25, 75])
    iqr_u = p75_u - p25_u

    median_du = np.median(du)
    p25_du, p75_du = np.percentile(du, [25, 75])
    iqr_du = p75_du - p25_du

    median_d2u = np.median(d2u)
    p25_d2u, p75_d2u = np.percentile(d2u, [25, 75])
    iqr_d2u = p75_d2u - p25_d2u

    # Normalized position of peak (on absolute value)
    norm_argmax_u = np.argmax(abs_u) / len(u_ac_buf)
    norm_argmax_du = np.argmax(abs_du) / len(du)
    norm_argmax_d2u = np.argmax(abs_d2u) / len(d2u)

    # Zero-crossing rate
    zcr_u = np.sum(np.diff(np.signbit(u_ac_buf))) / (len(u_ac_buf) - 1.0)  # signbit for robustness
    zcr_du = np.sum(np.diff(np.signbit(du))) / (len(du) - 1.0)
    zcr_d2u = np.sum(np.diff(np.signbit(d2u))) / (len(d2u) - 1.0)

    # Peak-to-peak on all signals
    ptp_u = np.ptp(u_ac_buf)
    ptp_du = np.ptp(du)
    ptp_d2u = np.ptp(d2u)

    # === Extended feature list ===
    # All original features are preserved in their relative positions/groups,
    # and new related features are inserted immediately after the corresponding original ones.
    feats = [
        # Location & spread (original means/stds + robust extensions)
        np.mean(u_ac_buf),
        np.std(u_ac_buf),
        median_u,
        p25_u,
        p75_u,
        iqr_u,
        mean_abs_u,                     # useful companion to mean

        np.mean(du),
        np.std(du),
        median_du,
        p25_du,
        p75_du,
        iqr_du,
        mean_abs_du,

        np.mean(d2u),
        np.std(d2u),
        median_d2u,
        p25_d2u,
        p75_d2u,
        iqr_d2u,
        mean_abs_d2u,

        # Peak values & positions (original raw max/argmax + absolute/normalized extensions)
        np.max(u_ac_buf),               # raw max (kept exactly as original)
        np.argmax(u_ac_buf),            # raw argmax (kept exactly as original)
        peak_u,                         # absolute peak
        norm_argmax_u,                  # normalized absolute argmax

        np.max(du),
        np.argmax(du),
        peak_du,
        norm_argmax_du,

        np.max(d2u),
        np.argmax(d2u),
        peak_d2u,
        norm_argmax_d2u,

        # Amplitude measures (original RMS/ptp for u + extensions to du/d2u)
        rms_u,                          # original RMS
        rms_du,
        rms_d2u,

        ptp_u,                          # original ptp
        ptp_du,
        ptp_d2u,

        # Shape / impulsiveness factors (new – grouped together)
        crest_u, crest_du, crest_d2u,
        shape_u, shape_du, shape_d2u,
        impulse_u, impulse_du, impulse_d2u,
        clearance_u, clearance_du, clearance_d2u,

        # Higher-order moments (original skew/kurt for u + extensions)
        skew_u,                         # original skew
        kurt_u,                         # original kurtosis
        skew_du,
        kurt_du,
        skew_d2u,
        kurt_d2u,

        # Additional cheap proxies
        zcr_u, zcr_du, zcr_d2u,
    ]

    return np.array(feats, dtype=np.float32)


def process_file(fname, y_final, N_force, h, c1, c2, c3, c4, c5, phi_dc, a):
    data, sr = sf.read(fname)
    new_sr = int(1e6)
    frac = new_sr / sr
    idxs = (np.arange(int(len(data)*frac)) / frac).astype(np.int64)
    signal = data[idxs]

    y0 = y_final.copy()
    u_ac_buf = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, signal)

    # extract features from full u_ac_buf
    features_full = features_extraction(u_ac_buf)

    # split u_ac_ buf into 10 sub-segments and extract features from each
    segments = np.split(u_ac_buf, 10)

    features_1 = features_extraction(segments[0])
    features_2 = features_extraction(segments[1])
    features_3 = features_extraction(segments[2])
    features_4 = features_extraction(segments[3])
    features_5 = features_extraction(segments[4])
    features_6 = features_extraction(segments[5])
    features_7 = features_extraction(segments[6])
    features_8 = features_extraction(segments[7])
    features_9 = features_extraction(segments[8])
    features_10 = features_extraction(segments[9])

    return features_full, features_1, features_2, features_3, features_4, features_5, features_6, features_7, features_8, features_9, features_10

    


def build_state_matrix(train_file_list_path, val_file_list_path, test_file_list_path, results_dir, a, u_dc, mu, f):
    # Load filenames
    train_filenames = np.loadtxt(train_file_list_path, dtype=str)
    val_filenames = np.loadtxt(val_file_list_path, dtype=str)
    test_filenames = np.loadtxt(test_file_list_path, dtype=str)

    # Combine for single parallel processing
    filenames = np.concatenate([train_filenames, val_filenames, test_filenames])

    # Simulation params
    alpha, omega_0, Q_0, tau, beta, gamma, R, kappa = 19.2, f*2*np.pi, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
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

    # results: list of tuples
    # (full, seg1, seg2, ..., seg10)
    results_per_segment = list(zip(*results))

    # Stack each into a state matrix
    state_matrices = [np.vstack(seg) for seg in results_per_segment]

    # -------------------------
    # Save results
    # -------------------------
    os.makedirs(results_dir, exist_ok=True)
    one_tenth_dir = os.path.join(results_dir, "one-tenth")
    os.makedirs(one_tenth_dir, exist_ok=True)

    # Full 1-second state matrix
    np.savez_compressed(
        f"{results_dir}/f-{int(f)}.npz",
        state_matrices[0]
    )

    # 10 × 0.1-second state matrices
    for seg_idx in range(1, 11):
        np.savez_compressed(
            f"{one_tenth_dir}/f-{int(f)}-{seg_idx}.npz",
            state_matrices[seg_idx]
        )



import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--f', type=float, required=True, help='Value of f to process')
args = parser.parse_args()

if __name__ == '__main__':

    a = 0.9
    mu = 1.0 
    u_dc = 1.0
    f = args.f

    # Paths
    train_files = '/scratch/almo2783/scratch/rayson/design1/5cities/train-filenames-5cities-rayson.csv'
    val_files = '/scratch/almo2783/scratch/rayson/design1/5cities/val-filenames-5cities-rayson.csv'
    test_files = '/scratch/almo2783/scratch/rayson/design1/5cities/test-filenames-barcelona-rayson.csv'

    # Results dir
    results_dir = f"/scratch/almo2783/scratch/ml-paper/multi-sens/5cities/results"
    os.makedirs(results_dir, exist_ok=True)

    build_state_matrix(train_files, val_files, test_files, results_dir, a, u_dc, mu, f)