import numpy as np
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from numba import njit


@njit(fastmath=True)
def rk4_step_inplace(y, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, k1, k2, k3, k4, y_temp):
    x, v, theta, u_ac = y
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)

    k1[0] = v
    k1[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta
    k1[2] = -beta * theta + (gamma / (R**2)) * min_term
    k1[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + 0.5 * h * k1[i]
    y_temp[0] = y[0] + 0.5 * h * k1[0]
    y_temp[1] = y[1] + 0.5 * h * k1[1]
    y_temp[2] = y[2] + 0.5 * h * k1[2]
    y_temp[3] = y[3] + 0.5 * h * k1[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)

    k2[0] = v
    k2[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta
    k2[2] = -beta * theta + (gamma / (R**2)) * min_term
    k2[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + 0.5 * h * k2[i]
    y_temp[0] = y[0] + 0.5 * h * k2[0]
    y_temp[1] = y[1] + 0.5 * h * k2[1]
    y_temp[2] = y[2] + 0.5 * h * k2[2]
    y_temp[3] = y[3] + 0.5 * h * k2[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)

    k3[0] = v
    k3[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta
    k3[2] = -beta * theta + (gamma / (R**2)) * min_term
    k3[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + h * k3[i]
    y_temp[0] = y[0] + h * k3[0]
    y_temp[1] = y[1] + h * k3[1]
    y_temp[2] = y[2] + h * k3[2]
    y_temp[3] = y[3] + h * k3[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)

    k4[0] = v
    k4[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta
    k4[2] = -beta * theta + (gamma / (R**2)) * min_term
    k4[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    y[0] += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y[1] += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    y[2] += (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    y[3] += (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])


@njit(fastmath=True)
def rk4_step_inplace_with_force(y, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, k1, k2, k3, k4, y_temp, f_x):
    x, v, theta, u_ac = y
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)
    
    k1[0] = v
    k1[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta + f_x
    k1[2] = -beta * theta + (gamma / (R**2)) * min_term
    k1[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + 0.5 * h * k1[i]
    y_temp[0] = y[0] + 0.5 * h * k1[0]
    y_temp[1] = y[1] + 0.5 * h * k1[1]
    y_temp[2] = y[2] + 0.5 * h * k1[2]
    y_temp[3] = y[3] + 0.5 * h * k1[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)
    
    k2[0] = v
    k2[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta + f_x
    k2[2] = -beta * theta + (gamma / (R**2)) * min_term
    k2[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + 0.5 * h * k2[i]
    y_temp[0] = y[0] + 0.5 * h * k2[0]
    y_temp[1] = y[1] + 0.5 * h * k2[1]
    y_temp[2] = y[2] + 0.5 * h * k2[2]
    y_temp[3] = y[3] + 0.5 * h * k2[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)
    
    k3[0] = v
    k3[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta + f_x
    k3[2] = -beta * theta + (gamma / (R**2)) * min_term
    k3[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y_temp[i] = y[i] + h * k3[i]
    y_temp[0] = y[0] + h * k3[0]
    y_temp[1] = y[1] + h * k3[1]
    y_temp[2] = y[2] + h * k3[2]
    y_temp[3] = y[3] + h * k3[3]


    x, v, theta, u_ac = y_temp
    temp = a * u_ac + u_dc
    min_term = min(temp * temp, u_max * u_max)
    
    k4[0] = v
    k4[1] = -(omega0 / Q0) * v - (omega0**2) * x + alpha * theta + f_x
    k4[2] = -beta * theta + (gamma / (R**2)) * min_term
    k4[3] = -u_ac / tau + kappa * v

    # for i in range(4):
    #     y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    y[0] += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y[1] += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    y[2] += (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    y[3] += (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])


@njit(fastmath=True)
def simulate_transient(N, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa):

    y  = np.zeros(4)
    y[0] = 1e-9
    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)

    for k in range(N):
        rk4_step_inplace(y, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, k1, k2, k3, k4, y_temp)

    return y


@njit(fastmath=True)
def simulate_with_force(y, N, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, f_ext):

    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    y_temp = np.zeros(4)
    buf_u_ac = np.empty(N)

    for k in range(N):
        f_x = f_ext[k]
        rk4_step_inplace_with_force(y, h, omega0, Q0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, k1, k2, k3, k4, y_temp, f_x)
        buf_u_ac[k] = y[3]

    return buf_u_ac


@njit(fastmath=True)
def calc_skew(x):
    n = x.shape[0]
    if n < 2:
        return 0.0
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    if m2 <= 0.0:
        return 0.0
    m3 = np.mean((x - mu) ** 3)
    return m3 / (m2 ** 1.5)


@njit(fastmath=True)
def calc_kurtosis(x):
    n = x.shape[0]
    if n < 2:
        return 0.0
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    if m2 <= 0.0:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return m4 / (m2 ** 2) - 3.0


@njit(fastmath=True)
def extract_features(u_ac_buf):
    # unchanged – already highly optimized Numba code
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
    skew_u = calc_skew(u_ac_buf)
    kurt_u = calc_kurtosis(u_ac_buf)
    skew_du = calc_skew(du)
    kurt_du = calc_kurtosis(du)
    skew_d2u = calc_skew(d2u)
    kurt_d2u = calc_kurtosis(d2u)
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
    zcr_u = np.sum(np.abs(np.diff(np.signbit(u_ac_buf)))) / (len(u_ac_buf) - 1.0) if len(u_ac_buf) > 1 else 0.0
    zcr_du = np.sum(np.abs(np.diff(np.signbit(du)))) / (len(du) - 1.0) if len(du) > 1 else 0.0
    zcr_d2u = np.sum(np.abs(np.diff(np.signbit(d2u)))) / (len(d2u) - 1.0) if len(d2u) > 1 else 0.0
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


def compute_transient(parameters):
    f, a, u_dc, alpha, Q_0, tau, beta, gamma, R, kappa, omega_0, mu = parameters
    h = 1e-6
    u_max = 1.0

    y_final = simulate_transient(50_000_000, h, omega_0, Q_0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa)
    return (y_final, h, omega_0, Q_0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa)


def process_one_file(fname, precomp_list):
    data, sr = sf.read(fname)
    
    new_len = 1_000_000
    frac = new_len / sr
    idxs_len = int(len(data) * frac)
    idxs = (np.arange(idxs_len) / frac).astype(np.int64)
    signal = data[idxs]

    signal = np.asarray(signal, dtype=np.float32)

    n_freq = len(precomp_list)
    file_features = np.empty((n_freq * 60,), dtype=np.float32)

    for i, (y_final, h, omega_0, Q_0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa) in enumerate(precomp_list):
        u_ac_buf = simulate_with_force(y_final, new_len, h, omega_0, Q_0, alpha, beta, gamma, R, a, u_dc, u_max, tau, kappa, signal)
        feats = extract_features(u_ac_buf)
        file_features[i*60:(i+1)*60] = feats

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
    X_eval_b  = np.hstack((X_eval,  np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)))

    W = ridge_closed_form(X_train_b, Y_train, lam)

    y_train_true = Y_train if Y_train.ndim == 1 else np.argmax(Y_train, axis=1)
    y_eval_true  = Y_eval  if Y_eval.ndim == 1  else np.argmax(Y_eval, axis=1)

    y_train_pred = X_train_b @ W
    y_train_hats = np.argmax(y_train_pred, axis=1)
    train_accuracy = np.mean(y_train_hats == y_train_true)

    y_eval_pred = X_eval_b @ W
    y_eval_hats = np.argmax(y_eval_pred, axis=1)

    accuracy  = accuracy_score(y_eval_true, y_eval_hats)
    
    results = np.array([lam, train_accuracy, accuracy], dtype=np.float64)
    
    return results



def calculate_a_crit(alpha, Q_0, tau, beta, gamma, R, kappa, omega_0, u_dc):

    a_crit = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

    return a_crit




sensor_params = {
    445: {
        "alpha": 12.5,
        "omega": 2796.0174616949157,
        "Q": 29.0,
        "tau": 0.0034,
        "beta": 294.11764705882354,
        "gamma": 2256560.0,
        "R": 13.4,
        "kappa": 147800.0,
    },
    490: {
        "alpha": 12.5,
        "omega": 3078.7608005179973,
        "Q": 33.0,
        "tau": 0.0039000000000000003,
        "beta": 256.4102564102564,
        "gamma": 1800000.0,
        "R": 13.4,
        "kappa": 162540.0,
    },
    591: {
        "alpha": 12.5,
        "omega": 3713.3625165431354,
        "Q": 45.0,
        "tau": 0.0039000000000000003,
        "beta": 256.4102564102564,
        "gamma": 1800000.0,
        "R": 13.4,
        "kappa": 214190.0,
    },
}


if __name__ == '__main__':

    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])

    train_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    train_filenames = np.loadtxt(train_files_path, dtype=str)
    val_filenames = np.loadtxt(val_files_path, dtype=str)
    filenames = np.concatenate([train_filenames, val_filenames])

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    u_dc = -0.1
    ratio = 0.8
    mu = 100.0
    f_values = np.array([445, 490, 591])

    # a_crits = np.empty(len(f_values), dtype=float)

    # for i, f in enumerate(f_values):

    #     if f not in sensor_params:
    #         raise ValueError(f"Unexpected frequency: {f}")

    #     p = sensor_params[f]

    #     a_crits[i] = calculate_a_crit(
    #         alpha=p["alpha"],
    #         Q_0=p["Q"],
    #         tau=p["tau"],
    #         beta=p["beta"],
    #         gamma=p["gamma"],
    #         R=p["R"],
    #         kappa=p["kappa"],
    #         omega_0=p["omega"],
    #         u_dc=u_dc,
    #     )

    a_crits = np.array([0.32, 0.45, 0.31])

    a_values = ratio * a_crits

    # create a tuple of parameters
    parameter_tuples = [
        (
            f,
            a_values[i],
            u_dc,
            sensor_params[f]["alpha"],
            sensor_params[f]["Q"],
            sensor_params[f]["tau"],
            sensor_params[f]["beta"],
            sensor_params[f]["gamma"],
            sensor_params[f]["R"],
            sensor_params[f]["kappa"],
            sensor_params[f]["omega"],
            mu,
        )
        for i, f in enumerate(f_values)
    ]

    # Parallel precomputation of transients
    precomp_list = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(compute_transient)(params) for params in parameter_tuples
    )

    # Parallel over files (each file handles all frequencies)
    results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
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
            X_val_std,  labels_val,
            lam
        )
        for lam in lambda_values
    )

    outputs_arr = np.vstack(outputs)

    lambda_grid = outputs_arr[:, 0]
    train_acc   = outputs_arr[:, 1] * 100
    val_acc     = outputs_arr[:, 2] * 100


    idx_best = np.argmax(val_acc)

    best_val    = val_acc[idx_best]
    best_train  = train_acc[idx_best]
    best_lambda = lambda_grid[idx_best]

    # target = results[2]
    print("\nRidge regression results on validation set:")
    print(f"best Lambda: {best_lambda}")
    print(f"Training acc: {best_train:2f} %")
    print(f"Validation acc: {best_val:.2f} %")