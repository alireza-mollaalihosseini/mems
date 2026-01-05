# import os
# import sys
# import numpy as np
# from numba import njit
# import matplotlib.pyplot as plt
# from joblib import Parallel, delayed


# @njit(fastmath=True)
# def hopf_rhs(y, lam, alpha, beta, force):
#     x, yv = y
#     r2 = x*x + yv*yv
    
#     dx = (lam + alpha*r2)*x - (1.0 + beta*r2)*yv + force
#     dy = (lam + alpha*r2)*yv + (1.0 + beta*r2)*x

#     return dx, dy


# @njit(fastmath=True)
# def rk4_step_hopf(y, h, lam, alpha, beta, force):
    
#     k1x, k1y = hopf_rhs(y, lam, alpha, beta, force)

#     ytemp0 = y[0] + 0.5*h*k1x
#     ytemp1 = y[1] + 0.5*h*k1y
#     k2x, k2y = hopf_rhs((ytemp0,ytemp1), lam, alpha, beta, force)

#     ytemp0 = y[0] + 0.5*h*k2x
#     ytemp1 = y[1] + 0.5*h*k2y
#     k3x, k3y = hopf_rhs((ytemp0,ytemp1), lam, alpha, beta, force)

#     ytemp0 = y[0] + h*k3x
#     ytemp1 = y[1] + h*k3y
#     k4x, k4y = hopf_rhs((ytemp0,ytemp1), lam, alpha, beta, force)

#     y[0] += (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
#     y[1] += (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)


# @njit(fastmath=True)
# def run_parameter_sweep(lam_values, h, N, record_len, alpha, beta):

#     results = np.zeros((len(lam_values), record_len))
    
#     y = np.zeros(2)
#     y[0] = 1e-9   # tiny initial displacement

#     for i in range(len(lam_values)):
#         lam = lam_values[i]

#         # transient burn-in
#         for k in range(N):
#             rk4_step_hopf(y, h, lam, alpha, beta, 0.0)

#             # record last segment
#             if k >= N - record_len:
#                 results[i, k-(N-record_len)] = y[0]

#         # perturb between runs
#         y[0] += 1e-10

#     return results


# # === parameters ===
# # Simulation params
# alpha = -1.0       # supercritical Hopf
# beta  = 0.3
# lam_values = np.linspace(-1.0, 1.0, 101)

# h = 1e-3
# T = 1000
# N = int(T/h)
# record_len = 5000

# results = run_parameter_sweep(lam_values, h, N, record_len, alpha, beta)

# # results_dir = f"/scratch/almo2783/scratch/ml-paper/nonlinearity/results-10fold"
# # os.makedirs(results_dir, exist_ok=True)


# # np.save(
# #             f"{results_dir}/hopf_bifurcation-alpha-{alpha:.2f}-beta-{beta:.2f}-h-{h:.1e}-T-{T}-rec_len-{record_len}.npy",
# #             results
# #         )


# plt.figure(figsize=(16,8))
# plt.plot(lam_values, amplitude, "o-", color="darkorange")
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()
# plt.savefig(f"bifurcation-h-{h:1.e}-T-{T}.png", dpi=300)
# plt.close()


import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


# ---------------- Hopf Oscillator ----------------

@njit(fastmath=True)
def hopf_rhs(x, y, lam, alpha, beta, force):
    r2 = x*x + y*y
    
    dx = (lam + alpha*r2)*x - (1.0 + beta*r2)*y + force
    dy = (lam + alpha*r2)*y + (1.0 + beta*r2)*x

    return dx, dy


@njit(fastmath=True)
def rk4_step_hopf(x, y, h, lam, alpha, beta, force):

    k1x, k1y = hopf_rhs(x, y, lam, alpha, beta, force)

    x2 = x + 0.5 * h * k1x
    y2 = y + 0.5 * h * k1y
    k2x, k2y = hopf_rhs(x2, y2, lam, alpha, beta, force)

    x3 = x + 0.5 * h * k2x
    y3 = y + 0.5 * h * k2y
    k3x, k3y = hopf_rhs(x3, y3, lam, alpha, beta, force)

    x4 = x + h * k3x
    y4 = y + h * k3y
    k4x, k4y = hopf_rhs(x4, y4, lam, alpha, beta, force)

    x += (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y += (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)

    return x, y


# ---------------- PARALLEL Sweep ----------------

@njit(parallel=True, fastmath=True)
def run_parameter_sweep_parallel(
        lam_values, h, N, record_len, alpha, beta
):
    nlam = len(lam_values)
    results = np.zeros((nlam, record_len), dtype=np.float64)

    # Parallel over lambda values
    for i in prange(nlam):

        lam = lam_values[i]

        # independent state per lambda
        x = 1e-9
        y = 0.0

        # integrate
        for k in range(N):

            x, y = rk4_step_hopf(
                x, y, h, lam, alpha, beta, 0.0
            )

            if k >= N - record_len:
                results[i, k-(N-record_len)] = x

    return results


# ---------------- RUN ----------------

alpha = -1.0
beta  = 0.3
lam_values = np.linspace(-1.0, 1.0, 101)

fs = 44100
h = 1 / fs # 1e-3
T = 3000.0
N = int(T * fs)
record_len = int(10.0 * fs)

results = run_parameter_sweep_parallel(
    lam_values, h, N, record_len, alpha, beta
)

np.save(f"results--fs-{int(fs)}-T-{int(T)}.npy", results)

# ---------------- ANALYSIS ----------------

# amplitude = np.sqrt(np.mean(results**2, axis=1))
peak_to_peak = results.max(axis=1) - results.min(axis=1)

plt.figure(figsize=(16,8))
# plt.plot(lam_values, amplitude, "o-", color="darkorange")
plt.plot(lam_values, peak_to_peak, "o-", color="darkorange")
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"bifurcation-h-{h:.1e}-T-{T}-record_len-{int(record_len)}.png", dpi=300)
plt.close()
