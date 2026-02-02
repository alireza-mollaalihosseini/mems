import sys
import numpy as np
from numba import njit
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.style.use("ggplot")

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

  results = np.zeros(500000)
  y  = np.zeros(4)
  y[0] = 1e-9
  k1 = np.zeros(4)
  k2 = np.zeros(4)
  k3 = np.zeros(4)
  k4 = np.zeros(4)
  y_temp = np.zeros(4)

  for k in range(N):
    rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)
    if k >= N - 500000:
      results[k - (N - 500000)] = y[0]
  return results


def single_simulation(u_dc, a_val, f):
    omega_0 = f * 2 * np.pi
    h = 1e-6 * omega_0

    alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
    u_max = 1.0

    T = 50.0 * omega_0
    t_rec = omega_0 * 3.0

    N = int(T / h)

    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)

    phi_dc = u_dc / u_max

    results = simulate_transient(
        N, h, c1, c2, c3, c4,
        phi_dc, a_val
    )

    const_tol = 1e-8

    if np.std(results) < const_tol:
        return 0

    peaks, _ = find_peaks(results)
    if len(peaks) == 0:
        return 0

    maxima = results[peaks]
    unique_maxima = np.unique(np.round(maxima, 4))

    return 1 if np.std(unique_maxima) < 0.01 else len(unique_maxima)


f_values = np.sort(np.array([43630, 44120, 42650, 45590,  6390, 44610, 23540, 45100,  6880,
   42160,  2960, 46080, 43140, 24030, 49510, 39220,  3940,  5410,
   38730, 37260,  3450, 25010, 20600, 20110,  5900, 41180,  4430,
    4920, 24520, 47060, 40200, 37750, 41670, 46570, 50000, 39710,
   38240, 36770, 48040, 48530,  7370, 18640, 49020,  2470, 47550,
   40690, 22070, 36280, 23050,  7860, 19620, 18150, 19130, 35300,
   21090, 28930, 21580, 25990,  8840, 11290,  8350,  9330,  1980,
   35790]))

f_values = f_values[f_values>=18000]
# f_values = [1980]
# a_values = np.linspace(-25, 25, 501)
a_values = np.linspace(-100, 250, 1001)
u_dc_values = np.linspace(0.01, 1, 100)



for f_val in f_values:

    extrema_count = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
        delayed(single_simulation)(u_dc, a_val, f_val)
        for u_dc in u_dc_values
        for a_val in a_values
    )

    extrema_counts = np.array(extrema_count).reshape(len(u_dc_values), len(a_values)).T  # shape: (a_values, u_dc_values)

    # Mask out zeros (keep them black)
    masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

    # Use colormap and set bad values (masked ones) to black
    cmap = plt.cm.cool.copy()
    cmap.set_bad(color='black')

    # --- Plotting with pcolormesh ---
    fig, ax = plt.subplots(figsize=(16, 8))

    U, A = np.meshgrid(u_dc_values, a_values)

    im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

    # Add discrete colorbar with ticks at integers
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of unique maxima", fontsize=20)

    # Increase colorbar tick fontsize
    cbar.ax.tick_params(labelsize=20)

    # Axis settings
    ax.set_ylim(a_values.min(), a_values.max())
    ax.set_xlabel("Bias Voltage (V)", fontsize=20)
    ax.set_ylabel("Feedback Strength", fontsize=20)
    
    # Increase axis tick fontsize
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(
        f"unique-maxima-f-{int(f_val)}.png",
        dpi=300
    )
    plt.close()