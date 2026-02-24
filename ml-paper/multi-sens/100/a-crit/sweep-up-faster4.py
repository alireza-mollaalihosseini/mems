import numpy as np
from numba import njit
from scipy.signal import find_peaks
from joblib import Parallel, delayed


@njit(fastmath=True)
def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
    # (your original rk4_step_inplace - unchanged)
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
def integrate_one_a(y, a, N, h, c1, c2, c3, c4, phi_dc, buffer):
    """Integrate ONE a-value and fill the 500k buffer. Only ~4 MB memory."""
    k1 = np.zeros(4, dtype=np.float64)
    k2 = np.zeros(4, dtype=np.float64)
    k3 = np.zeros(4, dtype=np.float64)
    k4 = np.zeros(4, dtype=np.float64)
    y_temp = np.zeros(4, dtype=np.float64)

    for k in range(N):
        rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)
        if k >= N - 500000:
            buffer[k - (N - 500000)] = y[0]


def simulate_for_u_dc(u_dc, a_values, f):
    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0

    T = 50.0 * omega_0
    N = int(T / h)

    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)

    phi_dc = u_dc / u_max
    noise = 1e-10 / (alpha * gamma * u_max**2 / (beta * R**2 * omega_0**2))

    const_tol = 1e-8

    col_results = np.zeros(len(a_values), dtype=int)
    y = np.zeros(4, dtype=np.float64)
    y[0] = 1e-9
    buffer = np.zeros(500000, dtype=np.float64)   # only this buffer → ~4 MB

    for i in range(len(a_values)):
        a = a_values[i]
        integrate_one_a(y, a, N, h, c1, c2, c3, c4, phi_dc, buffer)

        data = buffer
        if np.std(data) < const_tol:
            col_results[i] = 0
        else:
            peaks, _ = find_peaks(data)
            maxima = data[peaks]
            unique_maxima = np.unique(np.round(maxima, 4))
            col_results[i] = 1 if np.std(unique_maxima) < 0.01 else len(unique_maxima)

        y[0] += noise

    return col_results


# ====================== PARALLEL EXECUTION (safe now) ======================
a_values = np.linspace(0, 10000, 10001)
u_dc = 1.0
f_values = np.linspace(1000, 50000, 100, dtype=int)

save_path = "/scratch/almo2783/scratch/test/a-crit/extrems/"

def run_and_save(f_val):
    print(f"Starting f = {int(f_val)} Hz ...")
    extrema = simulate_for_u_dc(u_dc, a_values, f_val)
    filename = f"{save_path}extrema_counts_f_{int(f_val)}-u-dc-1.0.npy"
    np.save(filename, extrema)
    return filename


# === THIS IS THE PARALLEL PART ===
results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=1)(
    delayed(run_and_save)(f_val) for f_val in f_values
)

print("All simulations finished and saved!")