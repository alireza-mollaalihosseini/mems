import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import welch

# Parameters
T = 500.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

omega_0 = 53956.46373431294
Q_0     = 50.0
alpha   = 19.2
beta    = 1066.0
gamma   = 1.62e7
R       = 12.5
tau     = 0.001
kappa   = 0.602e6
u_dc    = 0.7

a_values = np.linspace(-3, 4, 1001)

c_f_values = np.array([-50.0, -48.0, -46.0, -44.0, -42.0, -40.0, -38.0, -36.0, -34.0, -32.0, -30.0,
                       -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0,  -8.0,
                        -6.0,  -4.0,  -2.0,   0.0,   2.0,   4.0,   6.0,   8.0,  10.0,  12.0,  14.0,
                        16.0,  18.0,  20.0,  22.0,  24.0,  26.0,  28.0,  30.0,  32.0,  34.0,  36.0,
                        38.0,  40.0,  42.0,  44.0,  46.0,  48.0,  50.0])

# Preallocate for multiple metrics
# extrema_counts = np.zeros((len(a_values), len(c_f_values)), dtype=int)
# sync_errors = np.zeros((len(a_values), len(c_f_values)))
# correlations = np.zeros((len(a_values), len(c_f_values)))
# phase_sync_indices = np.zeros((len(a_values), len(c_f_values)))
peak_freqs1 = np.zeros((len(a_values), len(c_f_values)))
peak_freqs2 = np.zeros((len(a_values), len(c_f_values)))
const_tol = 1e-8
# h = omega_0 * 1e-6
# fs = 1 / h
fs = 1.0 / 1e-6

# --- Function to process one u_dc ---
def process_c_f(j, c_f, a_values, const_tol, T, t_rec):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/coupled/deflections-up/'
        f'deflections-RK4-up-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-04-cf-{c_f}.npy',
        mmap_mode="r"
    )
    
    # col_extrema = np.zeros(len(a_values), dtype=int)
    # col_sync = np.zeros(len(a_values))
    # col_corr = np.zeros(len(a_values))
    # col_phase_sync = np.zeros(len(a_values))
    col_peak1 = np.zeros(len(a_values))
    col_peak2 = np.zeros(len(a_values))

    for i, a in enumerate(a_values):
        x1 = results_up[i, 0, :]
        x2 = results_up[i, 1, :]

        if np.std(x1) < const_tol or np.std(x2) < const_tol:
            # col_extrema[i] = 0
            # col_sync[i] = np.nan # 0.0  # Perfect sync in fixed point
            # col_corr[i] = np.nan # 1.0
            # col_phase_sync[i] = np.nan # 1.0
            col_peak1[i] = np.nan # 0.0
            col_peak2[i] = np.nan # 0.0
            continue

        # Take last part of signal
        x1 = x1[-50000:]
        x2 = x2[-50000:]

        # # Original: Unique maxima count on x1
        # peaks, _ = find_peaks(x1)
        # maxima = x1[peaks]
        # if len(maxima) == 0:
        #     col_extrema[i] = 0
        # else:
        #     unique_maxima = np.unique(maxima.round(decimals=4))
        #     if np.std(unique_maxima) < 0.01:
        #         col_extrema[i] = 1
        #     else:
        #         col_extrema[i] = len(unique_maxima)

        # # Sync error: Normalized mean absolute difference
        # sync_error = np.mean(np.abs(x1 - x2)) / np.std(x1)
        # col_sync[i] = sync_error

        # # Pearson correlation at lag 0 (detrend for better accuracy)
        # x1_d = x1 - np.mean(x1)
        # x2_d = x2 - np.mean(x2)
        # corr, _ = pearsonr(x1_d, x2_d)
        # col_corr[i] = corr

        # # New: Phase sync index via Hilbert transform (order parameter)
        # # Compute instantaneous phases
        # analytic_x1 = hilbert(x1)
        # analytic_x2 = hilbert(x2)
        # phase1 = np.angle(analytic_x1)
        # phase2 = np.angle(analytic_x2)
        # phase_diff = np.unwrap(phase1 - phase2)  # Unwrap to handle discontinuities
        # phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # Center to [-π, π]
        # # Kuramoto-like order parameter: |<exp(i φ_diff)>| (1 = perfect sync, 0 = uniform)
        # phase_sync = np.abs(np.mean(np.exp(1j * phase_diff)))
        # col_phase_sync[i] = phase_sync

        # peak frequencies
        freqs, psd = welch(x1, fs=fs, nperseg=min(4096, len(x1)))
        peak_idx = np.argmax(psd)
        if peak_idx == 0:
            col_peak1[i] = 0
        else:
            col_peak1[i] = freqs[peak_idx]

        freqs, psd = welch(x2, fs=fs, nperseg=min(4096, len(x2)))
        peak_idx = np.argmax(psd)
        if peak_idx == 0:
            col_peak2[i] = 0
        else:
            col_peak2[i] = freqs[peak_idx]

    return j, col_peak1, col_peak2

# --- Run in parallel ---
results = Parallel(n_jobs=-1, backend="threading", verbose=5)(
    delayed(process_c_f)(j, c_f, a_values, const_tol, T, t_rec)
    for j, c_f in enumerate(c_f_values)
)

# --- Reconstruct matrices ---
# extrema_counts = np.zeros((len(a_values), len(c_f_values)), dtype=int)
# sync_errors = np.zeros((len(a_values), len(c_f_values)))
# correlations = np.zeros((len(a_values), len(c_f_values)))
# phase_sync_indices = np.zeros((len(a_values), len(c_f_values)))
peak_freqs1 = np.zeros((len(a_values), len(c_f_values)))
peak_freqs2 = np.zeros((len(a_values), len(c_f_values)))

for j, col_peak1, col_peak2 in results:
    # extrema_counts[:, j] = col_extrema
    # sync_errors[:, j] = col_sync
    # correlations[:, j] = col_corr
    # phase_sync_indices[:, j] = col_phase_sync
    peak_freqs1[:, j] = col_peak1
    peak_freqs2[:, j] = col_peak2

# Save the maps
# np.save('sync_errors.npy', sync_errors)
# np.save('correlations.npy', correlations)
# np.save('phase_sync_indices.npy', phase_sync_indices)
np.save('peak_freqs1.npy', peak_freqs1)
np.save('peak_freqs2.npy', peak_freqs2)

# # Mask out zeros (keep them black)
# masked_data = np.ma.masked_where(phase_sync_indices == np.nan, phase_sync_indices)

# # Use colormap and set bad values (masked ones) to black
# cmap = plt.cm.viridis_r.copy()
# cmap.set_bad(color='black')

# # --- Plotting with pcolormesh ---
# fig, ax = plt.subplots(figsize=(12, 6))

# U, A = np.meshgrid(c_f_values, a_values)

# im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

# # Add colorbar
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("Phase Sync Index", fontsize=16, fontweight="bold")

# # Axis settings
# ax.set_ylim(a_values.min(), a_values.max())
# ax.set_xlabel("c_f", fontsize=16, fontweight="bold")
# ax.set_ylabel("a", fontsize=16, fontweight="bold")
# ax.set_title("Phase Sync Index", fontsize=18, fontweight="bold")

# plt.tight_layout()
# plt.savefig(
#     f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-sensor-1-last-50000-values-Phase_Sync_Index.png",
#     dpi=300
# )
# plt.close()


# # Mask out zeros (keep them black)
# masked_data = np.ma.masked_where(correlations == 0, correlations)

# # Use colormap and set bad values (masked ones) to black
# cmap = plt.cm.viridis_r.copy()
# cmap.set_bad(color='black')

# # --- Plotting with pcolormesh ---
# fig, ax = plt.subplots(figsize=(12, 6))

# U, A = np.meshgrid(c_f_values, a_values)

# im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

# # Add colorbar
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# # Axis settings
# ax.set_ylim(a_values.min(), a_values.max())
# ax.set_xlabel("c_f", fontsize=16, fontweight="bold")
# ax.set_ylabel("a", fontsize=16, fontweight="bold")
# ax.set_title("Unique maxima count (Upward)", fontsize=18, fontweight="bold")

# plt.tight_layout()
# plt.savefig(
#     f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-sensor-1-last-50000-values-correlations.png",
#     dpi=300
# )
# plt.close()