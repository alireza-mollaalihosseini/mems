import numpy as np
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from numba import njit

@njit(fastmath=True)
def rk4_step_inplace_coupled(y, h,
                             c1_1, c2_1, c3_1, c4_1,
                             c1_2, c2_2, c3_2, c4_2,
                             phi_dc, a1, a2, g,
                             k1, k2, k3, k4, y_temp):
    exsi1 = y[0]; etta1 = y[1]; psy1 = y[2]; phi1 = y[3]
    exsi2 = y[4]; etta2 = y[5]; psy2 = y[6]; phi2 = y[7]

    # k1
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k1[0] = etta1
    k1[1] = -c2_1 * etta1 - exsi1 + psy1
    k1[2] = -c1_1 * psy1 + c1_1 * min_term1
    k1[3] = -c3_1 * phi1 + c4_1 * etta1
    k1[4] = etta2
    k1[5] = -c2_2 * etta2 - exsi2 + psy2
    k1[6] = -c1_2 * psy2 + c1_2 * min_term2
    k1[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k2
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k2[0] = etta1
    k2[1] = -c2_1 * etta1 - exsi1 + psy1
    k2[2] = -c1_1 * psy1 + c1_1 * min_term1
    k2[3] = -c3_1 * phi1 + c4_1 * etta1
    k2[4] = etta2
    k2[5] = -c2_2 * etta2 - exsi2 + psy2
    k2[6] = -c1_2 * psy2 + c1_2 * min_term2
    k2[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k3
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k3[0] = etta1
    k3[1] = -c2_1 * etta1 - exsi1 + psy1
    k3[2] = -c1_1 * psy1 + c1_1 * min_term1
    k3[3] = -c3_1 * phi1 + c4_1 * etta1
    k3[4] = etta2
    k3[5] = -c2_2 * etta2 - exsi2 + psy2
    k3[6] = -c1_2 * psy2 + c1_2 * min_term2
    k3[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + h * k3[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k4
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k4[0] = etta1
    k4[1] = -c2_1 * etta1 - exsi1 + psy1
    k4[2] = -c1_1 * psy1 + c1_1 * min_term1
    k4[3] = -c3_1 * phi1 + c4_1 * etta1
    k4[4] = etta2
    k4[5] = -c2_2 * etta2 - exsi2 + psy2
    k4[6] = -c1_2 * psy2 + c1_2 * min_term2
    k4[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def rk4_step_inplace_with_force_coupled(y, h,
                                        c1_1, c2_1, c3_1, c4_1, c5_1,
                                        c1_2, c2_2, c3_2, c4_2, c5_2,
                                        phi_dc, a1, a2, g,
                                        k1, k2, k3, k4, y_temp, f_x):
    exsi1 = y[0]; etta1 = y[1]; psy1 = y[2]; phi1 = y[3]
    exsi2 = y[4]; etta2 = y[5]; psy2 = y[6]; phi2 = y[7]

    # k1
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k1[0] = etta1
    k1[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k1[2] = -c1_1 * psy1 + c1_1 * min_term1
    k1[3] = -c3_1 * phi1 + c4_1 * etta1
    k1[4] = etta2
    k1[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k1[6] = -c1_2 * psy2 + c1_2 * min_term2
    k1[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k1[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k2
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k2[0] = etta1
    k2[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k2[2] = -c1_1 * psy1 + c1_1 * min_term1
    k2[3] = -c3_1 * phi1 + c4_1 * etta1
    k2[4] = etta2
    k2[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k2[6] = -c1_2 * psy2 + c1_2 * min_term2
    k2[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + 0.5 * h * k2[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k3
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k3[0] = etta1
    k3[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k3[2] = -c1_1 * psy1 + c1_1 * min_term1
    k3[3] = -c3_1 * phi1 + c4_1 * etta1
    k3[4] = etta2
    k3[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k3[6] = -c1_2 * psy2 + c1_2 * min_term2
    k3[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y_temp[i] = y[i] + h * k3[i]
    exsi1 = y_temp[0]; etta1 = y_temp[1]; psy1 = y_temp[2]; phi1 = y_temp[3]
    exsi2 = y_temp[4]; etta2 = y_temp[5]; psy2 = y_temp[6]; phi2 = y_temp[7]

    # k4
    temp1 = a1 * phi1 + g * phi2 + phi_dc
    min_term1 = min(temp1 * temp1, 1.0)
    temp2 = a2 * phi2 - g * phi1 + phi_dc
    min_term2 = min(temp2 * temp2, 1.0)
    k4[0] = etta1
    k4[1] = -c2_1 * etta1 - exsi1 + psy1 + c5_1 * f_x
    k4[2] = -c1_1 * psy1 + c1_1 * min_term1
    k4[3] = -c3_1 * phi1 + c4_1 * etta1
    k4[4] = etta2
    k4[5] = -c2_2 * etta2 - exsi2 + psy2 + c5_2 * f_x
    k4[6] = -c1_2 * psy2 + c1_2 * min_term2
    k4[7] = -c3_2 * phi2 + c4_2 * etta2

    for i in range(8):
        y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(fastmath=True)
def simulate_transient_coupled(N, h,
                               c1_1, c2_1, c3_1, c4_1,
                               c1_2, c2_2, c3_2, c4_2,
                               phi_dc, a1, a2, g):
    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = 1e-9
    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    for k in range(N):
        rk4_step_inplace_coupled(y, h,
                                 c1_1, c2_1, c3_1, c4_1,
                                 c1_2, c2_2, c3_2, c4_2,
                                 phi_dc, a1, a2, g,
                                 k1, k2, k3, k4, y_temp)
    return y


@njit(fastmath=True)
def simulate_with_force_coupled(y, N, h,
                                c1_1, c2_1, c3_1, c4_1, c5_1,
                                c1_2, c2_2, c3_2, c4_2, c5_2,
                                phi_dc, a1, a2, g, f_ext):
    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)
    buf_u_ac1 = np.empty(N)
    buf_u_ac2 = np.empty(N)

    for k in range(N):
        f_x = f_ext[k]
        rk4_step_inplace_with_force_coupled(y, h,
                                            c1_1, c2_1, c3_1, c4_1, c5_1,
                                            c1_2, c2_2, c3_2, c4_2, c5_2,
                                            phi_dc, a1, a2, g,
                                            k1, k2, k3, k4, y_temp, f_x)
        buf_u_ac1[k] = y[3]
        buf_u_ac2[k] = y[7]

    return buf_u_ac1, buf_u_ac2


def extract_features(u_ac_buf):
    du = np.diff(u_ac_buf)
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

    sqrt_abs_mean_u = np.mean(np.sqrt(abs_u))
    sqrt_abs_mean_du = np.mean(np.sqrt(abs_du))
    sqrt_abs_mean_d2u = np.mean(np.sqrt(abs_d2u))

    clearance_u = peak_u / (sqrt_abs_mean_u ** 2) if sqrt_abs_mean_u > 0 else 0.0
    clearance_du = peak_du / (sqrt_abs_mean_du ** 2) if sqrt_abs_mean_du > 0 else 0.0
    clearance_d2u = peak_d2u / (sqrt_abs_mean_d2u ** 2) if sqrt_abs_mean_d2u > 0 else 0.0

    skew_u = skew(u_ac_buf)
    kurt_u = kurtosis(u_ac_buf)
    skew_du = skew(du)
    kurt_du = kurtosis(du)
    skew_d2u = skew(d2u)
    kurt_d2u = kurtosis(d2u)

    median_u = np.median(u_ac_buf)
    p25_u, p75_u = np.percentile(u_ac_buf, [25, 75])
    iqr_u = p75_u - p25_u

    median_du = np.median(du)
    p25_du, p75_du = np.percentile(du, [25, 75])
    iqr_du = p75_du - p25_du

    median_d2u = np.median(d2u)
    p25_d2u, p75_d2u = np.percentile(d2u, [25, 75])
    iqr_d2u = p75_d2u - p25_d2u

    norm_argmax_u = np.argmax(abs_u) / len(u_ac_buf)
    norm_argmax_du = np.argmax(abs_du) / len(du)
    norm_argmax_d2u = np.argmax(abs_d2u) / len(d2u)

    zcr_u = np.sum(np.diff(np.signbit(u_ac_buf))) / (len(u_ac_buf) - 1.0)
    zcr_du = np.sum(np.diff(np.signbit(du))) / (len(du) - 1.0)
    zcr_d2u = np.sum(np.diff(np.signbit(d2u))) / (len(d2u) - 1.0)

    ptp_u = np.ptp(u_ac_buf)
    ptp_du = np.ptp(du)
    ptp_d2u = np.ptp(d2u)

    feats = [
        np.mean(u_ac_buf), np.std(u_ac_buf), median_u, p25_u, p75_u, iqr_u, mean_abs_u,
        np.mean(du), np.std(du), median_du, p25_du, p75_du, iqr_du, mean_abs_du,
        np.mean(d2u), np.std(d2u), median_d2u, p25_d2u, p75_d2u, iqr_d2u, mean_abs_d2u,
        np.max(u_ac_buf), np.argmax(u_ac_buf), peak_u, norm_argmax_u,
        np.max(du), np.argmax(du), peak_du, norm_argmax_du,
        np.max(d2u), np.argmax(d2u), peak_d2u, norm_argmax_d2u,
        rms_u, rms_du, rms_d2u,
        ptp_u, ptp_du, ptp_d2u,
        crest_u, crest_du, crest_d2u,
        shape_u, shape_du, shape_d2u,
        impulse_u, impulse_du, impulse_d2u,
        clearance_u, clearance_du, clearance_d2u,
        skew_u, kurt_u, skew_du, kurt_du, skew_d2u, kurt_d2u,
        zcr_u, zcr_du, zcr_d2u,
    ]

    return np.array(feats, dtype=np.float32)


def compute_transient_pair(params, mu=1.0):
    f1, a1, f2, a2, u_dc = params
    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0

    # Sensor 1
    omega_0_1 = f1 * 2 * np.pi
    h1 = 1e-6 * omega_0_1
    l_0_1 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0_1**2)
    c1_1 = beta / omega_0_1
    c2_1 = 1 / Q_0
    c3_1 = 1 / (tau * omega_0_1)
    c4_1 = (kappa * l_0_1) / u_max
    c5_1 = mu / (l_0_1 * omega_0_1**2)

    # Sensor 2
    omega_0_2 = f2 * 2 * np.pi
    h2 = 1e-6 * omega_0_2
    l_0_2 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0_2**2)
    c1_2 = beta / omega_0_2
    c2_2 = 1 / Q_0
    c3_2 = 1 / (tau * omega_0_2)
    c4_2 = (kappa * l_0_2) / u_max
    c5_2 = mu / (l_0_2 * omega_0_2**2)

    phi_dc = u_dc / u_max
    h = min(h1, h2)          # more conservative (lower) time-step
    g = 0.5

    y_final = simulate_transient_coupled(50_000_000, h,
                                         c1_1, c2_1, c3_1, c4_1,
                                         c1_2, c2_2, c3_2, c4_2,
                                         phi_dc, a1, a2, g)
    return (y_final, h,
            c1_1, c2_1, c3_1, c4_1, c5_1,
            c1_2, c2_2, c3_2, c4_2, c5_2,
            phi_dc, a1, a2, g)


def process_one_file(fname, precomp_list):
    data, sr = sf.read(fname)

    new_len = 1_000_000
    frac = new_len / sr
    idxs_len = int(len(data) * frac)
    idxs = (np.arange(idxs_len) / frac).astype(np.int64)
    signal = data[idxs]

    signal = np.asarray(signal, dtype=np.float32)

    n_pairs = len(precomp_list)
    file_features = np.empty((n_pairs * 120,), dtype=np.float32)

    for i, precomp in enumerate(precomp_list):
        (y_final, h,
         c1_1, c2_1, c3_1, c4_1, c5_1,
         c1_2, c2_2, c3_2, c4_2, c5_2,
         phi_dc, a1, a2, g) = precomp

        u_ac_buf1, u_ac_buf2 = simulate_with_force_coupled(
            y_final, new_len, h,
            c1_1, c2_1, c3_1, c4_1, c5_1,
            c1_2, c2_2, c3_2, c4_2, c5_2,
            phi_dc, a1, a2, g, signal)

        feats1 = extract_features(u_ac_buf1)
        feats2 = extract_features(u_ac_buf2)

        start = i * 120
        file_features[start:start + 60] = feats1
        file_features[start + 60:start + 120] = feats2

    return file_features


# Ridge functions unchanged (good as-is)
def ridge_closed_form(X_train, Y_train, lam):
    n_features = X_train.shape[1]
    I = np.eye(n_features, dtype=X_train.dtype)
    A = X_train.T @ X_train + lam * I
    b = X_train.T @ Y_train
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def ridge_regression_fast(X_train, Y_train, X_eval, Y_eval, lam):
    X_train_b = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))
    X_eval_b = np.hstack((X_eval, np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)

    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true = Y_eval if Y_eval.ndim == 1 else np.argmax(Y_eval, axis=1)

    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy = accuracy_score(y_eval_true, y_eval_hats)

    results = np.array([lam, train_accuracy, accuracy], dtype=np.float64)

    return results


if __name__ == '__main__':

    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])

    train_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    train_filenames = np.loadtxt(train_files_path, dtype=str)
    val_filenames = np.loadtxt(val_files_path, dtype=str)
    filenames = np.concatenate([train_filenames, val_filenames])

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    u_dc = 0.5
    ratio = 0.3
    # mu = 1.0
    mu = 100.0
    f_values = np.linspace(1_000, 50_000, 100, dtype=int)
    a_crits = np.load(f"/scratch/almo2783/scratch/test/a-crit/a-crits/a-crit-u-dc-{u_dc:.1f}-more.npy")
    a_values = a_crits * ratio

    # Pairing: consecutive sensors (close frequencies → physically most sensible)
    # Change the line below if you later want a different pairing (e.g. first-last):
    # pair_indices = [(i, 99 - i) for i in range(50)]
    pair_indices = [(2 * i, 2 * i + 1) for i in range(50)]

    parameter_pairs = [
        (f_values[idx1], a_values[idx1], f_values[idx2], a_values[idx2], u_dc)
        for idx1, idx2 in pair_indices
    ]

    # Parallel precomputation of transients (now 50 coupled pairs)
    precomp_list = Parallel(n_jobs=64, backend="loky", verbose=1)(
        delayed(compute_transient_pair)(params) for params in parameter_pairs
    )

    # Parallel over files (each file handles all 100 sensors via 50 pairs)
    results = Parallel(n_jobs=64, backend="loky", verbose=1)(
        delayed(process_one_file)(fname, precomp_list) for fname in filenames
    )

    state_matrix = np.vstack(results)

    # Split (adjust indices based on your subsample split)
    train_state = state_matrix[:len(labels_train)]
    val_state = state_matrix[len(train_filenames):]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std = scaler.transform(val_state)

    # evaluate all lambdas (parallel)
    outputs = Parallel(
        n_jobs=64,
        verbose=1,
        backend="multiprocessing"
    )(
        delayed(ridge_regression_fast)(
            X_train_std, labels_train,
            X_val_std, labels_val,
            lam
        )
        for lam in lambda_values
    )

    outputs_arr = np.vstack(outputs)

    lambda_grid = outputs_arr[:, 0]
    train_acc = outputs_arr[:, 1] * 100
    val_acc = outputs_arr[:, 2] * 100

    idx_best = np.argmax(val_acc)

    best_val = val_acc[idx_best]
    best_train = train_acc[idx_best]
    best_lambda = lambda_grid[idx_best]

    print("\nRidge regression results on validation set:")
    print(f"best Lambda: {best_lambda}")
    print(f"Training acc: {best_train:2f} %")
    print(f"Validation acc: {best_val:.2f} %")