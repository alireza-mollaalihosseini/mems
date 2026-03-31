# import numpy as np
# from numba import njit
# from scipy.signal import find_peaks
# from joblib import Parallel, delayed


# @njit(fastmath=True)
# def rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp):
#     # k1
#     exsi, etta, psy, phi_ac = y
#     temp = a * phi_ac + phi_dc
#     min_term = min(temp * temp, 1.0)
#     k1[0] = etta
#     k1[1] = -c2 * etta - exsi + psy
#     k1[2] = -c1 * psy + c1 * min_term
#     k1[3] = -c3 * phi_ac + c4 * etta

#     # k2
#     for i in range(4):
#         y_temp[i] = y[i] + 0.5 * h * k1[i]
#     exsi, etta, psy, phi_ac = y_temp
#     temp = a * phi_ac + phi_dc
#     min_term = min(temp * temp, 1.0)
#     k2[0] = etta
#     k2[1] = -c2 * etta - exsi + psy
#     k2[2] = -c1 * psy + c1 * min_term
#     k2[3] = -c3 * phi_ac + c4 * etta

#     # k3
#     for i in range(4):
#         y_temp[i] = y[i] + 0.5 * h * k2[i]
#     exsi, etta, psy, phi_ac = y_temp
#     temp = a * phi_ac + phi_dc
#     min_term = min(temp * temp, 1.0)
#     k3[0] = etta
#     k3[1] = -c2 * etta - exsi + psy
#     k3[2] = -c1 * psy + c1 * min_term
#     k3[3] = -c3 * phi_ac + c4 * etta

#     # k4
#     for i in range(4):
#         y_temp[i] = y[i] + h * k3[i]
#     exsi, etta, psy, phi_ac = y_temp
#     temp = a * phi_ac + phi_dc
#     min_term = min(temp * temp, 1.0)
#     k4[0] = etta
#     k4[1] = -c2 * etta - exsi + psy
#     k4[2] = -c1 * psy + c1 * min_term
#     k4[3] = -c3 * phi_ac + c4 * etta

#     # update y
#     for i in range(4):
#         y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


# @njit(fastmath=True)
# def run_simulation(a_values, N, n_rec, h, c1, c2, c3, c4, phi_dc, noise):
#     results = np.zeros((len(a_values), 500_000))
#     y = np.zeros(4)
#     y[0] = 1e-9

#     k1 = np.zeros(4)
#     k2 = np.zeros(4)
#     k3 = np.zeros(4)
#     k4 = np.zeros(4)
#     y_temp = np.zeros(4)

#     for i in range(len(a_values)):
#         a = a_values[i]
#         for k in range(N):
#             rk4_step_inplace(y, h, c1, c2, c3, c4, phi_dc, a, k1, k2, k3, k4, y_temp)
#             if k >= N - 500_000:
#                 results[i, k - (N - 500_000)] = y[0]
#         y[0] += noise
#     return results


# def simulate_for_u_dc(u_dc, a_values, f):
#     omega_0 = f * 2 * np.pi
#     h = 1e-6 * omega_0

#     alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6
#     u_max = 1.0

#     T = 50.0 * omega_0
#     t_rec = omega_0 * 3.0

#     l_0 = alpha * gamma * u_max**2 / (beta * R**2 * omega_0**2)
#     noise = 1e-10 / l_0

#     N = int(T / h)
#     n_rec = int(t_rec / h)

#     c1 = beta / omega_0
#     c2 = 1 / Q_0
#     c3 = 1 / (tau * omega_0)
#     c4 = (alpha * gamma * kappa * u_max) / (beta * R**2 * omega_0**2)

#     phi_dc = u_dc / u_max

#     results = run_simulation(
#         a_values, N, n_rec, h,
#         c1, c2, c3, c4,
#         phi_dc, noise
#     )

#     const_tol = 1e-8
#     col_results = np.zeros(len(a_values), dtype=int)

#     for i in range(len(a_values)):
#         data = results[i]

#         if np.std(data) < const_tol:
#             col_results[i] = 0
#             continue

#         peaks, _ = find_peaks(data)
#         maxima = data[peaks]
#         unique_maxima = np.unique(maxima.round(4))

#         col_results[i] = 1 if np.std(unique_maxima) < 0.01 else len(unique_maxima)

#     return col_results


# a_values = np.linspace(0, 1500, 1501)
# u_dc_values = np.linspace(0.1, 1, 10)
# f_values = np.linspace(1000, 50000, 100, dtype=int)

# for f_val in f_values:

#     extrema_counts = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(
#         delayed(simulate_for_u_dc)(u_dc, a_values, f=f_val)
#         for u_dc in u_dc_values
#     )

#     extrema_counts = np.array(extrema_counts).T  # shape: (a_values, u_dc_values)

#     np.save(f"/scratch/almo2783/scratch/test/a-crit/extrems/extrema_counts_f_{int(f_val)}-more.npy", extrema_counts)


import numpy as np
from numba import njit, prange
import os
import time
# from numba import set_num_threads
# set_num_threads(64)


@njit(fastmath=True)
def rk4_step(y, h, c1, c2, c3, c4, phi_dc, a):
    """Perform one RK4 step using registers for speed."""
    ex, et, ps, pa = y[0], y[1], y[2], y[3]
    
    # Define derivatives locally to avoid overhead
    # returns: dex, det, dps, dpa
    
    # k1
    tmp1 = a * pa + phi_dc
    k1_ex = et
    k1_et = -c2*et - ex + ps
    k1_ps = -c1*ps + c1*min(tmp1*tmp1, 1.0)
    k1_pa = -c3*pa + c4*et
    
    # k2
    ex2, et2, ps2, pa2 = ex + 0.5*h*k1_ex, et + 0.5*h*k1_et, ps + 0.5*h*k1_ps, pa + 0.5*h*k1_pa
    tmp2 = a * pa2 + phi_dc
    k2_ex = et2
    k2_et = -c2*et2 - ex2 + ps2
    k2_ps = -c1*ps2 + c1*min(tmp2*tmp2, 1.0)
    k2_pa = -c3*pa2 + c4*et2
    
    # k3
    ex3, et3, ps3, pa3 = ex + 0.5*h*k2_ex, et + 0.5*h*k2_et, ps + 0.5*h*k2_ps, pa + 0.5*h*k2_pa
    tmp3 = a * pa3 + phi_dc
    k3_ex = et3
    k3_et = -c2*et3 - ex3 + ps3
    k3_ps = -c1*ps3 + c1*min(tmp3*tmp3, 1.0)
    k3_pa = -c3*pa3 + c4*et3
    
    # k4
    ex4, et4, ps4, pa4 = ex + h*k3_ex, et + h*k3_et, ps + h*k3_ps, pa + h*k3_pa
    tmp4 = a * pa4 + phi_dc
    k4_ex = et4
    k4_et = -c2*et4 - ex4 + ps4
    k4_ps = -c1*ps4 + c1*min(tmp4*tmp4, 1.0)
    k4_pa = -c3*pa4 + c4*et4

    # Update state
    y[0] += (h/6.0) * (k1_ex + 2*k2_ex + 2*k3_ex + k4_ex)
    y[1] += (h/6.0) * (k1_et + 2*k2_et + 2*k3_et + k4_et)
    y[2] += (h/6.0) * (k1_ps + 2*k2_ps + 2*k3_ps + k4_ps)
    y[3] += (h/6.0) * (k1_pa + 2*k2_pa + 2*k3_pa + k4_pa)

@njit(fastmath=True)
def get_unique_peaks_count(data, tol=1e-4):
    """Fast peak finding and unique counting in Numba."""
    peaks = []
    # Identify local maxima
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            val = round(data[i] / tol) * tol
            # Check uniqueness manually (Numba list friendly)
            unique = True
            for p in peaks:
                if abs(p - val) < 1e-6:
                    unique = False
                    break
            if unique:
                peaks.append(val)
    
    if len(peaks) == 0: return 0
    
    # Standard deviation check for unique maxima (stability check)
    p_arr = np.array(peaks)
    if np.std(p_arr) < 0.01:
        return 1
    return len(peaks)

@njit(parallel=True, fastmath=True)
def run_simulation_parallel(a_values, u_dc, N, h, c1, c2, c3, c4, noise):
    """Main parallel loop for a single f_val and u_dc."""
    n_a = len(a_values)
    col_results = np.zeros(n_a, dtype=np.int32)
    phi_dc = u_dc / 1.0 # u_max = 1.0
    
    for i in prange(n_a):
        a = a_values[i]
        y = np.array([1e-9, 0.0, 0.0, 0.0])
        # Only record what we need to save memory
        record_size = 500_000
        recorded_data = np.zeros(record_size)
        
        for k in range(N):
            rk4_step(y, h, c1, c2, c3, c4, phi_dc, a)
            if k >= (N - record_size):
                recorded_data[k - (N - record_size)] = y[0]
        
        # Stats and peak counting
        if np.std(recorded_data) < 1e-8:
            col_results[i] = 0
        else:
            col_results[i] = get_unique_peaks_count(recorded_data)
            
    return col_results

# --- Constants and Execution ---
a_values = np.linspace(0, 1_500, 15_001)
u_dc_values = np.linspace(0.1, 1, 10)[:-1]
# u_dc_values = np.array([0.5])
f_values = np.linspace(1_000, 50_000, 100, dtype=int)
# f_values = np.array([1_000])

start = time.time()

# Physics parameters
alpha, Q_0, tau, beta, gamma, R, kappa = 19.2, 500.0, 0.001, 1066.0, 1.62e7, 16.5, 0.602e6

for f_val in f_values:
    omega_0 = f_val * 2 * np.pi
    h = 1e-6 * omega_0
    T = 50.0 * omega_0
    N = int(T / h)
    
    c1 = beta / omega_0
    c2 = 1 / Q_0
    c3 = 1 / (tau * omega_0)
    c4 = (alpha * gamma * kappa) / (beta * R**2 * omega_0**2) # u_max = 1.0 simplified

    # Prepare matrix for current f_val
    extrema_matrix = np.zeros((len(a_values), len(u_dc_values)))

    for j, u_dc in enumerate(u_dc_values):
        print(f"Processing f={f_val}, u_dc={u_dc:.2f}")
        # This replaces the Parallel(delayed) call with Numba's internal threading
        extrema_matrix[:, j] = run_simulation_parallel(
            a_values, u_dc, N, h, c1, c2, c3, c4, 1e-10
        )

    # Save
    save_path = f"/scratch/almo2783/scratch/test/a-crit/extrems/extrema_counts_f_{f_val}-more-more.npy"
    np.save(save_path, extrema_matrix)
    