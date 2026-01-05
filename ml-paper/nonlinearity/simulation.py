import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from numba import njit
from joblib import Parallel, delayed
import os
import sys
from functools import partial

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

# ------------------- ONE SIMULATION TASK -------------------
def simulate_for_a(a, params, f_ext, output_dir):
    (
        h, c1, c2, c3, c4, c5, phi_dc, N_transient, N_force
    ) = params

    print(f"Running transient + forced simulation for a={a:.2f} ...")

    # 1️⃣ Transient simulation
    y_final = simulate_transient(N_transient, h, c1, c2, c3, c4, phi_dc, a)

    # 2️⃣ Forced simulation
    y0 = y_final.copy()
    u_ac = simulate_with_force(y0, N_force, h, c1, c2, c3, c4, c5, phi_dc, a, f_ext)

    # Save
    out_path = os.path.join(output_dir, f"u_ac_a_{a:.2f}.npz")
    np.savez_compressed(out_path, u_ac=u_ac)
    return out_path

# ------------------- MAIN FUNCTION -------------------
def simulate_parallel(file_list_path, output_dir, a_values, u_dc, mu, seed=42):
    np.random.seed(seed)
    filenames = np.loadtxt(file_list_path, dtype=str)
    sample_file = np.random.choice(filenames)
    print(f"Selected file: {sample_file}")

    # Load sound
    data, sr = sf.read(sample_file)

    # Physical parameters
    alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0
    u_max = 1.0
    h = 1e-6 * omega_0
    T = 50.0 * omega_0
    N_transient = int((0.1 * omega_0) / h)
    N_force = int((1.0 * omega_0) / h)

    l_0 = (alpha * gamma * u_max**2) / (beta * R**2 * omega_0**2)
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (kappa * l_0) / (u_max)
    c5 = mu / (l_0 * omega_0**2)
    phi_dc = u_dc / u_max

    # Prepare forcing function
    new_sr = int((1.0 * omega_0) / h)
    frac = new_sr / sr
    idxs = (np.arange(int(len(data) * frac)) / frac).astype(np.int64)
    idxs = idxs[idxs < len(data)]

    b_filt, a_filt = butter(4, 0.49 * sr, fs=sr, btype="low")
    signal = lfilter(b_filt, a_filt, data[idxs])

    f_ext = np.zeros(N_force)
    f_ext[:len(signal)] = signal

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "original_signal.npz"), signal=signal)

    # Simulation parameters (shared across jobs)
    params = (h, c1, c2, c3, c4, c5, phi_dc, N_transient, N_force)

    # ---------------- Parallel Execution ----------------
    results = Parallel(n_jobs=-1, backend='threading', verbose=10)(
        delayed(simulate_for_a)(a, params, f_ext, output_dir)
        for a in a_values
    )

    print(f"\n✅ Completed {len(results)} simulations. Results in {output_dir}")
    return results

# ------------------- RUN -------------------
if __name__ == "__main__":
    file_list_path = "/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv"
    output_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    # a_values = np.linspace(-2, 2, 101)
    a_values = np.linspace(-1, 1, 101)
    u_dc = 0.4
    mu = 1.0

    simulate_parallel(file_list_path, output_dir, a_values, u_dc, mu, seed=42)
