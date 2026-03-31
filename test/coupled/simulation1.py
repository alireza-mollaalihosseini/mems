import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from numba import njit

@njit(fastmath=True)
def rk4_step_coupled(
    y, h,
    c1, c2, c3, c4,
    phi_dc1, phi_dc2,
    A,              # 2x2 coupling matrix
    k1, k2, k3, k4,
    y_temp
):

    # ---------- helper function ----------
    def rhs(y, k):

        # unpack
        x1,v1,th1,phi1, x2,v2,th2,phi2 = y

        # coupled temperatures
        temp1 = A[0,0]*phi1 + A[0,1]*phi2 + phi_dc1
        temp2 = A[1,1]*phi2 + A[1,0]*phi1 + phi_dc2

        m1 = min(temp1*temp1, 1.0)
        m2 = min(temp2*temp2, 1.0)

        # ---- sensor 1 ----
        k[0] = v1
        k[1] = -c2*v1 - x1 + th1
        k[2] = -c1*th1 + c1*m1
        k[3] = -c3*phi1 + c4*v1

        # ---- sensor 2 ----
        k[4] = v2
        k[5] = -c2*v2 - x2 + th2
        k[6] = -c1*th2 + c1*m2
        k[7] = -c3*phi2 + c4*v2

    # RK4 stages
    rhs(y, k1)

    for i in range(8):
        y_temp[i] = y[i] + 0.5*h*k1[i]
    rhs(y_temp, k2)

    for i in range(8):
        y_temp[i] = y[i] + 0.5*h*k2[i]
    rhs(y_temp, k3)

    for i in range(8):
        y_temp[i] = y[i] + h*k3[i]
    rhs(y_temp, k4)

    for i in range(8):
        y[i] += (h/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])


@njit(fastmath=True)
def simulate_coupled_record(
    N, h,
    c1, c2, c3, c4,
    phi_dc1, phi_dc2,
    A
):

    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = 1e-9

    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    # --- buffers ---
    x1 = np.empty(N)
    x2 = np.empty(N)
    phi1 = np.empty(N)
    phi2 = np.empty(N)

    for k in range(N):

        rk4_step_coupled(
            y, h,
            c1, c2, c3, c4,
            phi_dc1, phi_dc2,
            A,
            k1, k2, k3, k4,
            y_temp
        )

        # store only observables
        x1[k] = y[0]
        x2[k] = y[4]
        phi1[k] = y[3]
        phi2[k] = y[7]

    return x1, x2, phi1, phi2


@njit(fastmath=True)
def simulate_coupled(
    N, h,
    c1, c2, c3, c4,
    phi_dc1, phi_dc2,
    A
):

    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = 1e-9   # small perturbation second sensor

    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    for _ in range(N):
        rk4_step_coupled(
            y, h,
            c1, c2, c3, c4,
            phi_dc1, phi_dc2,
            A,
            k1, k2, k3, k4,
            y_temp
        )

    return y


@njit(fastmath=True)
def rk4_step_coupled_force(
    y, h,
    c1, c2, c3, c4, c5,
    phi_dc1, phi_dc2,
    A,
    k1, k2, k3, k4, y_temp,
    f_x
):

    def rhs(state, out):

        x1,v1,th1,phi1, x2,v2,th2,phi2 = state

        # ---- coupling ----
        temp1 = A[0,0]*phi1 + A[0,1]*phi2 + phi_dc1
        temp2 = A[1,1]*phi2 + A[1,0]*phi1 + phi_dc2

        m1 = min(temp1*temp1, 1.0)
        m2 = min(temp2*temp2, 1.0)

        # sensor 1
        out[0] = v1
        out[1] = -c2*v1 - x1 + th1 + c5*f_x
        out[2] = -c1*th1 + c1*m1
        out[3] = -c3*phi1 + c4*v1

        # sensor 2
        out[4] = v2
        out[5] = -c2*v2 - x2 + th2 + c5*f_x
        out[6] = -c1*th2 + c1*m2
        out[7] = -c3*phi2 + c4*v2

    # RK4 stages
    rhs(y, k1)

    for i in range(8):
        y_temp[i] = y[i] + 0.5*h*k1[i]
    rhs(y_temp, k2)

    for i in range(8):
        y_temp[i] = y[i] + 0.5*h*k2[i]
    rhs(y_temp, k3)

    for i in range(8):
        y_temp[i] = y[i] + h*k3[i]
    rhs(y_temp, k4)

    for i in range(8):
        y[i] += (h/6.0)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])



@njit(fastmath=True)
def simulate_transient_coupled(
    y, N_trans, h,
    c1,c2,c3,c4,c5,
    phi_dc1,phi_dc2,
    A
):

    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    for _ in range(N_trans):
        rk4_step_coupled_force(
            y, h,
            c1,c2,c3,c4,c5,
            phi_dc1,phi_dc2,
            A,
            k1,k2,k3,k4,y_temp,
            0.0   # no forcing during transient
        )

    return y


@njit(fastmath=True)
def simulate_record_coupled(
    y,
    N_rec, h,
    c1,c2,c3,c4,c5,
    phi_dc1,phi_dc2,
    A,
    f_ext
):

    k1 = np.zeros(8)
    k2 = np.zeros(8)
    k3 = np.zeros(8)
    k4 = np.zeros(8)
    y_temp = np.zeros(8)

    phi1 = np.empty(N_rec)
    phi2 = np.empty(N_rec)

    for k in range(N_rec):

        rk4_step_coupled_force(
            y, h,
            c1,c2,c3,c4,c5,
            phi_dc1,phi_dc2,
            A,
            k1,k2,k3,k4,y_temp,
            f_ext[k]
        )

        phi1[k] = y[3]
        phi2[k] = y[7]

    return phi1, phi2


@njit(fastmath=True)
def simulate_full_coupled(
    N_trans, N_rec, h,
    c1,c2,c3,c4,c5,
    phi_dc1,phi_dc2,
    A,
    f_ext
):

    y = np.zeros(8)
    y[0] = 1e-9
    y[4] = -1e-9   # symmetry breaking

    simulate_transient_coupled(
        y, N_trans, h,
        c1,c2,c3,c4,c5,
        phi_dc1,phi_dc2,
        A
    )

    return simulate_record_coupled(
        y,
        N_rec, h,
        c1,c2,c3,c4,c5,
        phi_dc1,phi_dc2,
        A,
        f_ext
    )


def process_one_file(fname, f, u_dc, mu, N_trans, N_rec, a, g):

    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 50.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0
    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / u_max
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max  # per-frequency
    phi_dc1 = phi_dc
    phi_dc2 = phi_dc

    data, sr = sf.read(fname)
    
    new_len = N_rec
    frac = new_len / sr
    idxs_len = int(len(data) * frac)
    idxs = (np.arange(idxs_len) / frac).astype(np.int64)
    signal = data[idxs]

    signal = np.asarray(signal, dtype=np.float32)
    f_ext = signal
    A = np.array([[a, g],
                  [-g, -a]])

    phi1, phi2 = simulate_full_coupled(N_trans, N_rec, h, c1,c2,c3,c4,c5, phi_dc1,phi_dc2, A, f_ext)

    feats = np.concatenate([np.abs(np.fft.rfft(phi1))[:12_000], np.abs(np.fft.rfft(phi2))[:12_000]])

    return feats


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


if __name__ == '__main__':
    a = 0.5
    u_dc = 0.3
    # mu_values = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0])
    # mu_values = np.array([100_000.0, 1_000_000.0, 10_000_000.0, 100_000_000.0])
    mu = 1e4
    f = 8_000
    N_trans = 50_000_000
    N_rec = 1_000_000
    g = -0.6

    lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])

    train_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
    val_files_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

    train_filenames = np.loadtxt(train_files_path, dtype=str)
    val_filenames = np.loadtxt(val_files_path, dtype=str)
    filenames = np.concatenate([train_filenames, val_filenames])

    labels_train = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
    labels_val   = np.load("/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

    # Parallel over files (each file handles all frequencies)
    # results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
    #     delayed(process_one_file)(fname, f, u_dc, mu, N_trans, N_rec, a, g) for fname in filenames
    # )
    results = Parallel(n_jobs=64, prefer="threads", verbose=1)(
        delayed(process_one_file)(fname, f, u_dc, mu, N_trans, N_rec, a, g) for fname in filenames
    )

    state_matrix = np.vstack(results)

    # Split (adjust indices based on your subsample split)
    train_state = state_matrix[:len(labels_train)]
    val_state = state_matrix[len(train_filenames):]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_state)
    X_val_std = scaler.transform(val_state)

    # evaluate all lambdas (parallel)
    # outputs = Parallel(
    #     n_jobs=64,
    #     verbose=1,
    #     backend="multiprocessing"
    # )(
    #     delayed(ridge_regression_fast)(
    #         X_train_std, labels_train,
    #         X_val_std,  labels_val,
    #         lam
    #     )
    #     for lam in lambda_values
    # )
    outputs = Parallel(
        n_jobs=64,
        verbose=1,
        prefer="threads"
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
    # best_lambda = lambdas[idx_best]
    best_lambda = lambda_grid[idx_best]

    # target = results[2]
    print("\nRidge regression results on validation set:")
    print(f"best Lambda: {best_lambda}")
    print(f"Training acc: {best_train:2f} %")
    print(f"Validation acc: {best_val:.2f} %")